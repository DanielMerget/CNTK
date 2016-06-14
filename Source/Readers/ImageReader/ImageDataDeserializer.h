//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "DataDeserializerBase.h"
#include "Config.h"
#include "ByteReader.h"
#include "ImageConfigHelper.h"
#include <unordered_map>
#include <numeric>

#include <inttypes.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// Used to keep track of the image. Accessed only using DenseSequenceData interface.
struct DeserializedImage : DenseSequenceData
{
    cv::Mat m_image;
};

// Image data deserializer based on the OpenCV library.
// The deserializer currently supports two output streams only: a feature and a label stream.
// All sequences consist only of a single sample (image/label).
// For features it uses dense storage format with different layout (dimensions) per sequence.
// For labels it uses either csc sparse format (classification) or dense format (regression).
template<LabelType labelType, class PrecisionType>
class ImageDataDeserializer : public DataDeserializerBase
{
public:
    explicit ImageDataDeserializer(const ConfigParameters& config)
    {
        ImageConfigHelper configHelper(config);
        m_streams = configHelper.GetStreams();
        assert(m_streams.size() == 2);
        m_grayscale = configHelper.UseGrayscale();
        const auto& label = m_streams[configHelper.GetLabelStreamId()];
        const auto& feature = m_streams[configHelper.GetFeatureStreamId()];

        // Expect data in HWC.
        ImageDimensions dimensions(*feature->m_sampleLayout, configHelper.GetDataFormat());
        feature->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));

        // could be done statically via a template, but not as simple and readable
        label->m_storageType = labelType == LabelType::Classification ? StorageType::sparse_csc : StorageType::dense;
        feature->m_storageType = StorageType::dense;

        m_labelDimension = label->m_sampleLayout->GetDim(0);
        if (m_labelDimension > numeric_limits<IndexType>::max())
        {
            RuntimeError("Label dimension (%" PRIu64 ") exceeds the maximum allowed "
                "value (%" PRIu64 ")\n", m_labelDimension, (size_t)numeric_limits<IndexType>::max());
        }

        CreateSequenceDescriptions(configHelper.GetMapPath(), configHelper);
    }

    // Gets sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
    virtual ChunkPtr GetChunk(size_t chunkId) override
    {
        auto sequenceDescription = m_imageSequences[chunkId];
        return std::make_shared<ImageChunk<labelType,PrecisionType>>(sequenceDescription, *this);
    }

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override
    {
        ChunkDescriptions result;
        result.reserve(m_imageSequences.size());
        for (auto const& s : m_imageSequences)
        {
            auto chunk = std::make_shared<ChunkDescription>();
            chunk->m_id = s.m_chunkId;
            chunk->m_numberOfSamples = 1;
            chunk->m_numberOfSequences = 1;
            result.push_back(chunk);
        }

        return result;
    }

    // Gets sequence descriptions for the chunk.
    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& result) override
    {
        // Currently a single sequence per chunk.
        result.push_back(m_imageSequences[chunkId]);
    }

private:
    // Image sequence descriptions. Currently, a sequence contains a single sample only.
    // Class not defined to prevent instantiation of non-specialized versions
    template<LabelType labelType, class PrecisionType>
    struct ImageSequenceDescription; // : public SequenceDescription

    // Specialized template for classification
    template<class PrecisionType>
    struct ImageSequenceDescription<LabelType::Classification, PrecisionType> : public SequenceDescription
    {
        std::string m_path;
        size_t m_classId;
    };

    // Specialized template for regression
    template<class PrecisionType>
    struct ImageSequenceDescription<LabelType::Regression, PrecisionType> : public SequenceDescription
    {
        std::string m_path;
        std::vector<PrecisionType> m_label;
    };

    template<LabelType labelType>
    void parseLine(const std::string& line, const size_t lineIndex, const std::string& mapPath, ImageSequenceDescription<labelType, PrecisionType>& description) const;

    template<>
    void parseLine(const std::string& line, const size_t lineIndex, const std::string& mapPath, ImageSequenceDescription<LabelType::Classification, PrecisionType>& description) const
    {
        std::stringstream ss(line);
        std::string imagePath;
        std::string read;

        if (!std::getline(ss, imagePath, '\t') || !std::getline(ss, read, '\t'))
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());

        char* eptr;
        errno = 0;
        description.m_classId = strtoull(read.c_str(), &eptr, 10);
        if (read.c_str() == eptr || errno == ERANGE)
            RuntimeError("Cannot parse label value on line %" PRIu64 ", second column, in file %s.", lineIndex, mapPath.c_str());

        if (description.m_classId >= m_labelDimension)
        {
            RuntimeError(
                "Image '%s' has invalid class id '%" PRIu64 "'. Expected label dimension is '%" PRIu64 "'. Line %" PRIu64 " in file %s.",
                imagePath.c_str(), description.m_classId, m_labelDimension, lineIndex, mapPath.c_str());
        }

        description.m_path = imagePath;
    };

    template<>
    void parseLine(const std::string& line, const size_t lineIndex, const std::string& mapPath, ImageSequenceDescription<LabelType::Regression, PrecisionType>& description) const
    {
        std::stringstream ss(line);
        std::string imagePath;
        std::string read;
        PrecisionType value;
        std::vector<PrecisionType> result;

        if (!std::getline(ss, imagePath, '\t'))
            RuntimeError("Could not read map file, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());

        for (size_t i = 0; i < m_labelDimension; ++i)
        {
            auto invoke_error = [=]()
            {
                RuntimeError("Could not parse label value on line %" PRIu64 ", column %d, in file %s.", lineIndex, i+1, mapPath.c_str());
            };

            if (!std::getline(ss, read, '\t'))
                invoke_error();
            char* eptr;
            errno = 0;
            value = static_cast<PrecisionType>(strtod(read.c_str(), &eptr));
            if (read.c_str() == eptr || errno == ERANGE)
            {
                // try to recover to std::nan
                if (strcmp(read.c_str(), "nan"))
                    value = std::numeric_limits<PrecisionType>::lowest();
                else
                    invoke_error();
            }

            result.push_back(value);
        }

        description.m_path = imagePath;
        description.m_label = result;
    };

    // Creates a set of sequence descriptions.
    void CreateSequenceDescriptions(std::string mapPath, const ImageConfigHelper& config)
    {
        std::ifstream mapFile(mapPath);
        if (!mapFile)
        {
            RuntimeError("Could not open %s for reading.", mapPath.c_str());
        }

        size_t itemsPerLine = config.IsMultiViewCrop() ? 10 : 1;
        size_t curId = 0;
        std::string line;
        PathReaderMap knownReaders;
        ImageSequenceDescription<labelType,PrecisionType> description;

        description.m_numberOfSamples = 1;
        description.m_isValid = true;

        for (size_t lineIndex = 0; std::getline(mapFile, line); ++lineIndex)
        {
            parseLine(line, lineIndex, mapPath, description);
            std::string imagePath = description.m_path;
            
            for (size_t start = curId; curId < start + itemsPerLine; curId++)
            {
                description.m_id = curId;
                description.m_chunkId = curId;
                description.m_path = imagePath;
                description.m_key.m_sequence = description.m_id;
                description.m_key.m_sample = 0;

                m_imageSequences.push_back(description);
                RegisterByteReader(description.m_id, description.m_path, knownReaders);
            }
        }
    }

    // For image, chunks correspond to a single image.
    template<LabelType labelType, class PrecisionType>
    class ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk<labelType, PrecisionType>>
    {
        ImageSequenceDescription<labelType, PrecisionType> m_description;
        ImageDataDeserializer<labelType, PrecisionType>& m_parent;

        template<LabelType labelType>
        SequenceDataPtr createLabeledSequence();

        // Specialization for classification
        template<>
        SequenceDataPtr createLabeledSequence<LabelType::Classification>()
        {
            auto result = std::make_shared<SparseSequenceData>();
            m_parent.CreateLabelFor(m_description, *result);
            return result;
        }

        // Specialization for regression
        template<>
        SequenceDataPtr createLabeledSequence<LabelType::Regression>()
        {
            auto result = std::make_shared<DenseSequenceData>();
            m_parent.CreateLabelFor(m_description, *result);
            return result;
        }

    public:
        ImageChunk(const ImageSequenceDescription<labelType, PrecisionType>& description, ImageDataDeserializer<labelType, PrecisionType>& parent)
            : m_description(description), m_parent(parent)
        {
        }

        virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
        {
            assert(sequenceId == m_description.m_id);

            auto image = std::make_shared<DeserializedImage>();
            image->m_image = std::move(m_parent.ReadImage(m_description.m_id, m_description.m_path, m_parent.m_grayscale));
            auto& cvImage = image->m_image;

            if (!cvImage.data)
            {
                RuntimeError("Cannot open file '%s'", m_description.m_path.c_str());
            }

            // could be done statically via a template, but not as simple and readable
            const int dataType = std::is_same<PrecisionType, float>::value ? CV_32F : CV_64F;
            if (cvImage.type() != CV_MAKETYPE(dataType, cvImage.channels()))
            {
                cvImage.convertTo(cvImage, dataType);
            }

            if (!cvImage.isContinuous())
            {
                cvImage = cvImage.clone();
            }
            assert(cvImage.isContinuous());

            image->m_data = image->m_image.data;
            ImageDimensions dimensions(cvImage.cols, cvImage.rows, cvImage.channels());
            image->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
            image->m_id = sequenceId;
            image->m_numberOfSamples = 1;
            image->m_chunk = shared_from_this();

            SequenceDataPtr label = createLabeledSequence<labelType>();
            label->m_chunk = shared_from_this();
            label->m_numberOfSamples = 1;

            result.push_back(image);
            result.push_back(label);
        }
    };

    template<LabelType labelType, class SequenceType>
    void CreateLabelFor(ImageSequenceDescription<labelType, PrecisionType>& desc, SequenceType& data);

    // Specialization for classification
    template<>
    void ImageDataDeserializer::CreateLabelFor(ImageSequenceDescription<LabelType::Classification, PrecisionType>& desc, SparseSequenceData& data)
    {
        auto zero_to_n = [](size_t n)
        {
            std::vector<IndexType> x(n);
            std::iota(x.begin(), x.end(), 0);
            return x; 
        };
        static PrecisionType one(1);
        static std::vector<IndexType> indices = zero_to_n(m_labelDimension);
        data.m_nnzCounts.resize(1);
        data.m_nnzCounts[0] = 1;
        data.m_totalNnzCount = 1;
        data.m_data = &one;
        data.m_indices = &(indices[desc.m_classId]);
    }

    // Specialization for regression
    template<>
    void CreateLabelFor(ImageSequenceDescription<LabelType::Regression, PrecisionType>& desc, DenseSequenceData& data)
    {
        data.m_data = static_cast<void*>(desc.m_label.data());
    }

    // whether images shall be loaded in grayscale 
    bool m_grayscale;
    size_t m_labelDimension;
    
    // Sequence descriptions for all input data.
    std::vector<ImageSequenceDescription<labelType, PrecisionType>> m_imageSequences;

    // Not using nocase_compare here as it's not correct on Linux.
    using PathReaderMap = std::unordered_map<std::string, std::shared_ptr<ByteReader>>;
    
    void RegisterByteReader(size_t seqId, const std::string& path, PathReaderMap& knownReaders)
    {
        assert(!path.empty());

        auto atPos = path.find_first_of('@');
        // Is it container or plain image file?
        if (atPos == std::string::npos)
            return;
        // REVIEW alexeyk: only .zip container support for now.
#ifdef USE_ZIP
        assert(atPos > 0);
        assert(atPos + 1 < path.length());
        auto containerPath = path.substr(0, atPos);
        // skip @ symbol and path separator (/ or \)
        auto itemPath = path.substr(atPos + 2);
        // zlib only supports / as path separator.
        std::replace(begin(itemPath), end(itemPath), '\\', '/');
        std::shared_ptr<ByteReader> reader;
        auto r = knownReaders.find(containerPath);
        if (r == knownReaders.end())
        {
            reader = std::make_shared<ZipByteReader>(containerPath);
            knownReaders[containerPath] = reader;
        }
        else
        {
            reader = (*r).second;
        }
        reader->Register(seqId, itemPath);
        m_readers[seqId] = reader;
#else
        UNUSED(seqId);
        UNUSED(knownReaders);
        RuntimeError("The code is built without zip container support. Only plain image files are supported.");
#endif
    }
    
    cv::Mat ReadImage(size_t seqId, const std::string& path, bool grayscale)
    {
        assert(!path.empty());

        ImageDataDeserializer::SeqReaderMap::const_iterator r;
        if (m_readers.empty() || (r = m_readers.find(seqId)) == m_readers.end())
            return m_defaultReader.Read(seqId, path, grayscale);
        return (*r).second->Read(seqId, path, grayscale);
    }

    // REVIEW alexeyk: can potentially use vector instead of map. Need to handle default reader and resizing though.
    using SeqReaderMap = std::unordered_map<size_t, std::shared_ptr<ByteReader>>;
    SeqReaderMap m_readers;

    FileByteReader m_defaultReader;
};

}}}

//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <limits>
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

cv::Mat FileByteReader::Read(size_t, const std::string& path, bool grayscale)
{
    assert(!path.empty());

    return cv::imread(path, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}

// instantiate templates explicitely
template class ImageDataDeserializer<LabelType::Classification, float>;
template class ImageDataDeserializer<LabelType::Classification, double>;
template class ImageDataDeserializer<LabelType::Regression, float>;
template class ImageDataDeserializer<LabelType::Regression, double>;

}}}

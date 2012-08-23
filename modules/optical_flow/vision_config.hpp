#ifndef VISION_CONFIG_HPP
#define VISION_CONFIG_HPP

#pragma once

#ifdef _MSC_VER
#define VISION_EXTERN_TEMPLATE template
#else // !defined(_MSC_VER)
#define VISION_EXTERN_TEMPLATE extern template
#endif // _MSC_VER

#endif // VISION_CONFIG_HPP

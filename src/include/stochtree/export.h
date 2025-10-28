/*!
 * Export macros ensure that the C++ code can be used as a library cross-platform 
 * (declspec needed to load names from a DLL on windows) and can be wrapped in a 
 * C program.
 * 
 * This code modifies (changing names of) the export macros in LightGBM, which carries 
 * the following copyright information:
 * 
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_EXPORT_H_
#define STOCHTREE_EXPORT_H_

/** Macros for exporting symbols in MSVC/GCC/CLANG **/

#ifdef __cplusplus
#define STOCHTREE_EXTERN_C extern "C"
#else
#define STOCHTREE_EXTERN_C
#endif

#ifdef _MSC_VER
#define STOCHTREE_EXPORT __declspec(dllexport)
#define STOCHTREE_C_EXPORT STOCHTREE_EXTERN_C __declspec(dllexport)
#else
#define STOCHTREE_EXPORT  __attribute__ ((visibility ("default")))
#define STOCHTREE_C_EXPORT STOCHTREE_EXTERN_C  __attribute__ ((visibility ("default")))
#endif

#endif /** STOCHTREE_EXPORT_H_ **/

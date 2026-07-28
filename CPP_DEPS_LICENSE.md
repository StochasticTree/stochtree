stochtree
Copyright 2023-2025 stochtree contributors

Several stochtree C++ header and source files include or are inspired by code 
in several open-source decision tree libraries: xgboost, LightGBM, and treelite. 
Copyright and license information for each of these three projects are detailed 
further below and in comments in each of the files.
File: include/stochtree/category_tracker.h [xgboost]
File: include/stochtree/common.h [xgboost]
File: include/stochtree/ensemble.h [xgboost]
File: include/stochtree/io.h [LightGBM]
File: include/stochtree/log.h [LightGBM]
File: include/stochtree/meta.h [LightGBM]
File: include/stochtree/partition_tracker.h [LightGBM, xgboost]
File: include/stochtree/tree.h [xgboost, treelite]

This project includes software from the xgboost project (Apache, 2.0).
* Copyright 2015-2024, XGBoost Contributors

This project includes software from the LightGBM project (MIT).
* Copyright (c) 2016 Microsoft Corporation

This project includes software from the treelite project (Apache, 2.0).
* Copyright (c) 2017-2023 by [treelite] Contributors

This project includes software from the fast_double_parser project (Apache, 2.0).
* Copyright (c) Daniel Lemire

This project includes software from the JSON for Modern C++ project (MIT).
* Copyright © 2013-2025 Niels Lohmann

This project includes software from the Eigen project (MPL, 2.0), whose headers carry the following copyrights:
File: deps/eigen/Eigen/Core
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/SparseCholesky
Copyright (C) 2008-2013 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/SparseLU
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/StdDeque
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/StdList
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/StdVector
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/src/Cholesky/LDLT.h
Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Keir Mierle <mierle@gmail.com>
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2011 Timothy E. Holy <tim.holy@gmail.com >

File: deps/eigen/Eigen/src/Cholesky/LLT_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Cholesky/LLT.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/CholmodSupport/CholmodSupport.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/ArithmeticSequence.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Array.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/ArrayBase.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/ArrayWrapper.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Assign_MKL.h
Copyright (c) 2011, Intel Corporation. All rights reserved.
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Assign.h
Copyright (C) 2007 Michael Olbrich <michael.olbrich@gmx.net>
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/AssignEvaluator.h
Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2011-2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Core/BandMatrix.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Block.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/BooleanRedux.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/CommaInitializer.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/ConditionEstimator.h
Copyright (C) 2016 Rasmus Munk Larsen (rmlarsen@google.com)

File: deps/eigen/Eigen/src/Core/CoreEvaluators.h
Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2011-2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Core/CoreIterators.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/CwiseBinaryOp.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/CwiseNullaryOp.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/CwiseTernaryOp.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>

File: deps/eigen/Eigen/src/Core/CwiseUnaryOp.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/CwiseUnaryView.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/DenseBase.h
Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/DenseCoeffsBase.h
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/DenseStorage.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2010-2013 Hauke Heibel <hauke.heibel@gmail.com>

File: deps/eigen/Eigen/src/Core/Diagonal.h
Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/DiagonalMatrix.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/DiagonalProduct.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Dot.h
Copyright (C) 2006-2008, 2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/EigenBase.h
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/ForceAlignedAccess.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Fuzzy.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/GeneralProduct.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/GenericPacketMath.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/GlobalFunctions.h
Copyright (C) 2010-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/IndexedView.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Inverse.h
Copyright (C) 2014-2019 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/IO.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Map.h
Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/MapBase.h
Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/MathFunctions.h
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

File: deps/eigen/Eigen/src/Core/MathFunctionsImpl.h
Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Matrix.h
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/MatrixBase.h
Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/NestByValue.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/NoAlias.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/NumTraits.h
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/PartialReduxEvaluator.h
Copyright (C) 2011-2018 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/PermutationMatrix.h
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/PlainObjectBase.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Product.h
Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/ProductEvaluators.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Core/Random.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Redux.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Ref.h
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Replicate.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Reshaped.h
Copyright (C) 2008-2017 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2014 yoco <peter.xiau@gmail.com>

File: deps/eigen/Eigen/src/Core/ReturnByValue.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Reverse.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009 Ricard Marxer <email@ricardmarxer.com>
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Select.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/SelfAdjointView.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/SelfCwiseBinaryOp.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Solve.h
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/SolverBase.h
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/SolveTriangular.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/StableNorm.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/StlIterators.h
Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Stride.h
Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Swap.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Transpose.h
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/Transpositions.h
Copyright (C) 2010-2011 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/TriangularMatrix.h
Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/VectorBlock.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/VectorwiseOp.h
Copyright (C) 2008-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/Visitor.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/AltiVec/Complex.h
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010-2016 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/AltiVec/MathFunctions.h
Copyright (C) 2007 Julien Pommier
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/AltiVec/MatrixProduct.h
Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
Copyright (C) 2021 Chip Kerchner (chip.kerchner@ibm.com)

File: deps/eigen/Eigen/src/Core/arch/AltiVec/MatrixProductMMA.h
Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
Copyright (C) 2021 Chip Kerchner (chip.kerchner@ibm.com)

File: deps/eigen/Eigen/src/Core/arch/AltiVec/PacketMath.h
Copyright (C) 2008-2016 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/AVX/Complex.h
Copyright (C) 2014 Benoit Steiner (benoit.steiner.goog@gmail.com)

File: deps/eigen/Eigen/src/Core/arch/AVX/MathFunctions.h
Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)

File: deps/eigen/Eigen/src/Core/arch/AVX/PacketMath.h
Copyright (C) 2014 Benoit Steiner (benoit.steiner.goog@gmail.com)

File: deps/eigen/Eigen/src/Core/arch/AVX/TypeCasting.h
Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>

File: deps/eigen/Eigen/src/Core/arch/AVX512/Complex.h
Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/AVX512/MathFunctions.h
Copyright (C) 2016 Pedro Gonnet (pedro.gonnet@gmail.com)

File: deps/eigen/Eigen/src/Core/arch/AVX512/PacketMath.h
Copyright (C) 2016 Benoit Steiner (benoit.steiner.goog@gmail.com)

File: deps/eigen/Eigen/src/Core/arch/AVX512/TypeCasting.h
Copyright (C) 2019 Rasmus Munk Larsen <rmlarsen@google.com>

File: deps/eigen/Eigen/src/Core/arch/CUDA/Complex.h
Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
Copyright (C) 2021 C. Antonio Sanchez <cantonios@google.com>

File: deps/eigen/Eigen/src/Core/arch/Default/BFloat16.h
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

File: deps/eigen/Eigen/src/Core/arch/Default/ConjHelper.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h
Copyright (C) 2007 Julien Pommier
Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
Copyright (C) 2009-2019 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/Default/GenericPacketMathFunctionsFwd.h
Copyright (C) 2019 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/Default/Half.h
Copyright (c) Fabian Giesen, 2016.

File: deps/eigen/Eigen/src/Core/arch/Default/Half.h
Copyright (c) Fabian Giesen, 2016

File: deps/eigen/Eigen/src/Core/arch/Default/Settings.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/arch/Default/TypeCasting.h
Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
Copyright (C) 2019 Rasmus Munk Larsen <rmlarsen@google.com>

File: deps/eigen/Eigen/src/Core/arch/GPU/MathFunctions.h
Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>

File: deps/eigen/Eigen/src/Core/arch/GPU/PacketMath.h
Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>

File: deps/eigen/Eigen/src/Core/arch/GPU/TypeCasting.h
Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>

File: deps/eigen/Eigen/src/Core/arch/MSA/Complex.h
Copyright (C) 2018 Wave Computing, Inc.

File: deps/eigen/Eigen/src/Core/arch/MSA/MathFunctions.h
Copyright (C) 2007 Julien Pommier
Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
File: deps/eigen/Eigen/src/Core/arch/MSA/MathFunctions.h
Copyright (C) 2018 Wave Computing, Inc.

File: deps/eigen/Eigen/src/Core/arch/MSA/PacketMath.h
Copyright (C) 2018 Wave Computing, Inc.

File: deps/eigen/Eigen/src/Core/arch/NEON/Complex.h
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/NEON/PacketMath.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/NEON/TypeCasting.h
Copyright (C) 2018 Rasmus Munk Larsen <rmlarsen@google.com>
Copyright (C) 2020 Antonio Sanchez <cantonios@google.com>

File: deps/eigen/Eigen/src/Core/arch/SSE/Complex.h
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/SSE/MathFunctions.h
Copyright (C) 2007 Julien Pommier
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/SSE/PacketMath.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/arch/SSE/TypeCasting.h
Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>

File: deps/eigen/Eigen/src/Core/arch/SVE/MathFunctions.h
Copyright (C) 2020, Arm Limited and Contributors

File: deps/eigen/Eigen/src/Core/arch/SVE/PacketMath.h
Copyright (C) 2020, Arm Limited and Contributors

File: deps/eigen/Eigen/src/Core/arch/SVE/TypeCasting.h
Copyright (C) 2020, Arm Limited and Contributors

File: deps/eigen/Eigen/src/Core/arch/SYCL/SyclMemoryModel.h
Copyright (C) 2017 Codeplay Software Limited

File: deps/eigen/Eigen/src/Core/arch/ZVector/Complex.h
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/ZVector/MathFunctions.h
Copyright (C) 2007 Julien Pommier
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/arch/ZVector/PacketMath.h
Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>

File: deps/eigen/Eigen/src/Core/functors/AssignmentFunctors.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/functors/BinaryFunctors.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/functors/NullaryFunctors.h
Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/functors/StlFunctors.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/functors/TernaryFunctors.h
Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>

File: deps/eigen/Eigen/src/Core/functors/UnaryFunctors.h
Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/GeneralBlockPanelKernel.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/GeneralMatrixMatrix_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/GeneralMatrixMatrix.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/GeneralMatrixMatrixTriangular_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/GeneralMatrixMatrixTriangular.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/GeneralMatrixVector_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/GeneralMatrixVector.h
Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/Parallelizer.h
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/SelfadjointMatrixMatrix_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/SelfadjointMatrixMatrix.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/SelfadjointMatrixVector_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/SelfadjointMatrixVector.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/SelfadjointProduct.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/SelfadjointRank2Update.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/TriangularMatrixMatrix_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/TriangularMatrixMatrix.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/TriangularMatrixVector_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/TriangularMatrixVector.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/TriangularSolverMatrix_BLAS.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/products/TriangularSolverMatrix.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/products/TriangularSolverVector.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/BlasUtil.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/ConfigureVectorization.h
Copyright (C) 2008-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2020, Arm Limited and Contributors

File: deps/eigen/Eigen/src/Core/util/Constants.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2020, Arm Limited and Contributors

File: deps/eigen/Eigen/src/Core/util/ForwardDeclarations.h
Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/IndexedViewHelper.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/IntegralConstant.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/Macros.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/util/Memory.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009 Kenneth Riddile <kfriddile@yahoo.com>
Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
Copyright (C) 2010 Thomas Capricelli <orzel@freehackers.org>
Copyright (C) 2013 Pavel Holoborodko <pavel@holoborodko.com>

File: deps/eigen/Eigen/src/Core/util/Meta.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/util/MKL_support.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Core/util/ReshapedHelper.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/StaticAssert.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Core/util/SymbolicIndex.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Core/util/XprHelper.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Eigenvalues/ComplexEigenSolver.h
Copyright (C) 2009 Claire Maurice
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/ComplexSchur_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Eigenvalues/ComplexSchur.h
Copyright (C) 2009 Claire Maurice
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/EigenSolver.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/GeneralizedEigenSolver.h
Copyright (C) 2012-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
Copyright (C) 2016 Tobias Wood <tobias@spinicist.org.uk>

File: deps/eigen/Eigen/src/Eigenvalues/GeneralizedSelfAdjointEigenSolver.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/HessenbergDecomposition.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/MatrixBaseEigenvalues.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/RealQZ.h
Copyright (C) 2012 Alexey Korepanov <kaikaikai@yandex.ru>

File: deps/eigen/Eigen/src/Eigenvalues/RealSchur_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Eigenvalues/RealSchur.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/SelfAdjointEigenSolver_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Eigenvalues/Tridiagonalization.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>

File: deps/eigen/Eigen/src/Geometry/AlignedBox.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/AlignedBox.h
Copyright (c) 2011-2014, Willow Garage, Inc.
Copyright (c) 2014-2015, Open Source Robotics Foundation

File: deps/eigen/Eigen/src/Geometry/AngleAxis.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/EulerAngles.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/Homogeneous.h
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/Hyperplane.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Geometry/OrthoMethods.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Geometry/ParametrizedLine.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/Geometry/Quaternion.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Mathieu Gautier <mathieu.gautier@cea.fr>

File: deps/eigen/Eigen/src/Geometry/Rotation2D.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/RotationBase.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/Scaling.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/Transform.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>

File: deps/eigen/Eigen/src/Geometry/Translation.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Geometry/Umeyama.h
Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>

File: deps/eigen/Eigen/src/Geometry/arch/Geometry_SIMD.h
Copyright (C) 2009 Rohit Garg <rpg.314@gmail.com>
Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Householder/BlockHouseholder.h
Copyright (C) 2010 Vincent Lejeune
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Householder/Householder.h
Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Householder/HouseholderSequence.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/BasicPreconditioners.h
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/BiCGSTAB.h
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/ConjugateGradient.h
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/IncompleteCholesky.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/IncompleteLUT.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/IncompleteLUT.h
Copyright (C) 2005, the Regents of the University of Minnesota

File: deps/eigen/Eigen/src/IterativeLinearSolvers/IterativeSolverBase.h
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/LeastSquareConjugateGradient.h
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/IterativeLinearSolvers/SolveWithGuess.h
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/Jacobi/Jacobi.h
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/KLUSupport/KLUSupport.h
Copyright (C) 2017 Kyle Macfarlan <kyle.macfarlan@gmail.com>

File: deps/eigen/Eigen/src/LU/Determinant.h
Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/LU/FullPivLU.h
Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/LU/InverseImpl.h
Copyright (C) 2008-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/LU/PartialPivLU_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/LU/PartialPivLU.h
Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/LU/arch/InverseSize4.h
Copyright (C) 2001 Intel Corporation
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/LU/arch/InverseSize4.h
Copyright (c) 2001 Intel Corporation.

File: deps/eigen/Eigen/src/MetisSupport/MetisSupport.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/misc/Image.h
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/misc/Kernel.h
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/misc/lapacke.h
Copyright (c) 2010, Intel Corp.

File: deps/eigen/Eigen/src/misc/RealSvd2x2.h
Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2013-2016 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/OrderingMethods/Amd.h
Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/OrderingMethods/Amd.h
Copyright (c) 2006, Timothy A. Davis.

File: deps/eigen/Eigen/src/OrderingMethods/Eigen_Colamd.h
Copyright (C) 2012 Desire Nuentsa Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/OrderingMethods/Eigen_Colamd.h
Copyright (c) 1998-2003 by the University of Florida.

File: deps/eigen/Eigen/src/OrderingMethods/Eigen_Colamd.h
Copyright, this License, and the

File: deps/eigen/Eigen/src/OrderingMethods/Ordering.h
Copyright (C) 2012  Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/PardisoSupport/PardisoSupport.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/PaStiXSupport/PaStiXSupport.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/plugins/BlockMethods.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/plugins/CommonCwiseBinaryOps.h
Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/plugins/CommonCwiseUnaryOps.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/plugins/IndexedViewMethods.h
Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/plugins/MatrixCwiseBinaryOps.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/plugins/MatrixCwiseUnaryOps.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/QR/ColPivHouseholderQR_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/QR/ColPivHouseholderQR.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/QR/CompleteOrthogonalDecomposition.h
Copyright (C) 2016 Rasmus Munk Larsen <rmlarsen@google.com>

File: deps/eigen/Eigen/src/QR/FullPivHouseholderQR.h
Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>

File: deps/eigen/Eigen/src/QR/HouseholderQR_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/QR/HouseholderQR.h
Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2010 Vincent Lejeune

File: deps/eigen/Eigen/src/SparseCholesky/SimplicialCholesky_impl.h
Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCholesky/SimplicialCholesky_impl.h
Copyright (c) 2005 by Timothy A. Davis.  All Rights Reserved.

File: deps/eigen/Eigen/src/SparseCholesky/SimplicialCholesky.h
Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/AmbiVector.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/CompressedStorage.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/ConservativeSparseSparseProduct.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/MappedSparseMatrix.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseAssign.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseBlock.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseColEtree.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseColEtree.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseCore/SparseCompressedBase.h
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseCwiseBinaryOp.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseCwiseUnaryOp.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseDenseProduct.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseDiagonalProduct.h
Copyright (C) 2009-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseDot.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseFuzzy.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseMap.h
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseMatrix.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseMatrixBase.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparsePermutation.h
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseProduct.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseRedux.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseRef.h
Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseSelfAdjointView.h
Copyright (C) 2009-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseSolverBase.h
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseSparseProductWithPruning.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseTranspose.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseTriangularView.h
Copyright (C) 2009-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseUtil.h
Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseVector.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseCore/SparseView.h
Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2010 Daniel Lowengrub <lowdanie@gmail.com>

File: deps/eigen/Eigen/src/SparseCore/TriangularSolver.h
Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_column_bmod.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_column_bmod.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_column_dfs.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_column_dfs.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_copy_to_ucol.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_copy_to_ucol.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_heap_relax_snode.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_heap_relax_snode.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_kernel_bmod.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_Memory.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_Memory.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_panel_bmod.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_panel_bmod.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_panel_dfs.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_panel_dfs.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_pivotL.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_pivotL.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_pruneL.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_pruneL.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_relax_snode.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_relax_snode.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SparseLU/SparseLU_Structs.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_SupernodalMatrix.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU_Utils.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLU.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SparseLU/SparseLUImpl.h
Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>

File: deps/eigen/Eigen/src/SparseQR/SparseQR.h
Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SPQRSupport/SuiteSparseQRSupport.h
Copyright (C) 2012 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/StlSupport/details.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/src/StlSupport/StdDeque.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/src/StlSupport/StdList.h
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/src/StlSupport/StdVector.h
Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>

File: deps/eigen/Eigen/src/SuperLUSupport/SuperLUSupport.h
Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SuperLUSupport/SuperLUSupport.h
Copyright (c) 1994 by Xerox Corporation.  All rights reserved.

File: deps/eigen/Eigen/src/SVD/BDCSVD.h
Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
Copyright (C) 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
Copyright (C) 2014-2017 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SVD/JacobiSVD_LAPACKE.h
Copyright (c) 2011, Intel Corporation. All rights reserved.

File: deps/eigen/Eigen/src/SVD/JacobiSVD.h
Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/SVD/SVDBase.h
Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>
File: deps/eigen/Eigen/src/SVD/SVDBase.h
Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>

File: deps/eigen/Eigen/src/SVD/UpperBidiagonalization.h
Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>

File: deps/eigen/Eigen/src/UmfPackSupport/UmfPackSupport.h
Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>

This project includes software from the fmt project, released under the following license and copyright
Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--- Optional exception to the license ---

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into a machine-executable object form of such
source code, you may redistribute such embedded portions in such object form
without including the above copyright and permission notices.


//
// Created by Lsc2001 on 2022/3/24.
//

#ifndef HSEGBENCHMARK_TORCH_H
#define HSEGBENCHMARK_TORCH_H

#ifdef slots
#undef slots
#endif

#include <torch/torch.h>
#include <torch/script.h>

#define slots Q_SLOTS

#endif //HSEGBENCHMARK_TORCH_H

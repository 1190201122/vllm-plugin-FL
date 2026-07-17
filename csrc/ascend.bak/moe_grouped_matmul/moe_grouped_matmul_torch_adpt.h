/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MOE_GROUPED_MATMUL_TORCH_ADPT_H
#define MOE_GROUPED_MATMUL_TORCH_ADPT_H

namespace vllm_fl {

std::vector<at::Tensor> moe_grouped_matmul(
    at::Tensor x,
    at::Tensor weight,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type)
{
    bool transpose_weight = false;
    bool weight_nz = true;

    at::TensorList x_list = at::TensorList(x);
    at::TensorList weight_list = at::TensorList(weight);
    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_list[0].options().dtype(x[0].scalar_type());
    auto m = x_list[0].sizes()[0];
    auto n = weight_list[0].sizes()[1];
    if (!transpose_weight) {
        n = weight_list[0].sizes()[2];
    }
    at::Tensor y_0 = at::empty(at::IntArrayRef{m, n}, options);
    y.emplace_back(y_0);
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnMoeGroupedMatmulWeightNz,
                 x_list, weight_list, group_list, transpose_weight, result);

    return y;
}

}
#endif

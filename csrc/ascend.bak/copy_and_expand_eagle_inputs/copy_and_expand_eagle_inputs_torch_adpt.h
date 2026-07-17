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
#ifndef COPY_AND_EXPAND_EAGLE_INPUTS_TORCH_ADPT_H
#define COPY_AND_EXPAND_EAGLE_INPUTS_TORCH_ADPT_H

namespace vllm_fl {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_copy_and_expand_eagle_inputs(
    const at::Tensor &target_token_ids,
    const at::Tensor &target_positions,
    const at::Tensor &next_token_ids,
    const at::Tensor &query_start_loc,
    const at::Tensor &query_end_loc,
    int64_t padding_token_id,
    int64_t parallel_drafting_token_id,
    int64_t num_padding_slots_per_request,
    bool shift_input_ids,
    int64_t total_draft_tokens)
{
    int64_t total_input_tokens = target_token_ids.size(0);
    int64_t num_reqs = query_start_loc.size(0) - 1;

    auto device = target_token_ids.device();
    at::Tensor out_input_ids = at::empty({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_positions = at::empty({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_is_rejected_token_mask = at::empty({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_is_masked_token_mask = at::empty({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_new_token_indices = at::empty({num_reqs * num_padding_slots_per_request}, at::dtype(at::kInt).device(device));
    at::Tensor out_hidden_state_mapping = at::empty({total_input_tokens}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnCopyAndExpandEagleInputs,
                 target_token_ids, target_positions, next_token_ids, query_start_loc, query_end_loc,
                 padding_token_id, parallel_drafting_token_id, num_padding_slots_per_request,
                 shift_input_ids, total_input_tokens,
                 out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
                 out_new_token_indices, out_hidden_state_mapping);

    return {out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
            out_new_token_indices, out_hidden_state_mapping};
}

}
#endif

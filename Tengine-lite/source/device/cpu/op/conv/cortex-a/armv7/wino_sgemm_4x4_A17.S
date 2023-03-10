/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
 */

// x0        output         v12-v15
// x1        input          v8
// x2        kernel         v0
// x3        cin

// x4       cin/4
// x3       cin_resi_4

    .section .text, "ax"
    .align 5

    .type wino_sgemm_4x4_A17 STT_FUNC
    .global wino_sgemm_4x4_A17
    .hidden wino_sgemm_4x4_A17

wino_sgemm_4x4_A17:
    push        {r4, lr}

    vmov.i64    q12, #0x0
    vmov.i64    q13, #0x0
    vmov.i64    q14, #0x0
    vmov.i64    q15, #0x0

    cmp        r3, #0x4
    blt        loop4_end
    lsr        r4, r3, #0x2

loop4:
    vldr        d0,  [r2]
    vldr        d16, [r1]
    vldr        d17, [r1, #0x8]
    vldr        d1,  [r2, #0x8]
    subs        r4, r4, #0x1
    vmla.f32    q12, q8, d0[0]
    vldr        d18, [r1, #0x10]
    vmla.f32    q13, q8, d0[1]
    vldr        d19, [r1, #0x18]
    vmla.f32    q14, q8, d1[0]
    vldr        d2,  [r2, #0x10]
    vmla.f32    q15, q8, d1[1]
    vldr        d3,  [r2, #0x18]
    vmla.f32    q12, q9, d2[0]
    vldr        d20, [r1, #0x20]
    vmla.f32    q13, q9, d2[1]
    vldr        d21, [r1, #0x28]
    vmla.f32    q14, q9, d3[0]
    vldr        d4,  [r2, #0x20]
    vmla.f32    q15, q9, d3[1]
    vldr        d5,  [r2, #0x28]
    vmla.f32    q12, q10,d4[0]
    vldr        d22, [r1, #0x30]
    vmla.f32    q13, q10,d4[1]
    vldr        d23, [r1, #0x38]
    vmla.f32    q14, q10,d5[0]
    vldr        d6,  [r2, #0x30]
    vmla.f32    q15, q10,d5[1]
    vldr        d7,  [r2, #0x38]
    vmla.f32    q12, q11,d6[0]
    vmla.f32    q13, q11,d6[1]
    pld        [r2, #0x180]
    add        r2, r2, #0x40
    vmla.f32    q14, q11,d7[0]
    pld        [r1, #0x180]
    add        r1, r1, #0x40
    vmla.f32    q15, q11,d7[1]
    bne        loop4

loop4_end:
    ands        r3, r3, #0x3
    beq        save_result

loop1:
    vldm        r1!, {d16 - d17}    // i[3-0]0
    vldm        r2!, {d0  -  d1}    // k[3-0]0
    subs        r3, r3, #0x1
    vmla.f32    q12, q8, d0[0]
    vmla.f32    q13, q8, d0[1]
    vmla.f32    q14, q8, d1[0]
    vmla.f32    q15, q8, d1[1]
    bne        loop1

save_result:
    vstm        r0, {d24-d31}

end:
    pop        {r4,pc}

    .end

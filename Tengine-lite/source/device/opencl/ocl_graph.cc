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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#include "ocl_graph.hpp"
#include "ocl_executor.hpp"

extern "C" {
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
}

int ocl_dev_init(struct device* dev)
{
    (void)dev;
    return 0;
}

int ocl_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    subgraph->device_graph = new OCLEngine;
    auto engine = (OCLEngine*)subgraph->device_graph;

    return engine->OCLEnginePreRun(subgraph);
}

int ocl_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (OCLEngine*)subgraph->device_graph;
    return engine->OCLEngineRun(subgraph);
}

int ocl_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (OCLEngine*)subgraph->device_graph;
    engine->OCLEnginePostRun();
    delete engine;

    return 0;
}

int ocl_dev_release(struct device* dev)
{
    (void)dev;
    return 0;
}

/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.shims

import com.nvidia.spark.rapids._

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.trees.TreePattern._

object SparkShimImpl extends Spark330PlusShims with Spark321PlusDBShims {
  // AnsiCast is removed from Spark3.4.0
  override def ansiCastRule: ExprRule[_ <: Expression] = null
}

trait ShimGetArrayStructFields extends ExtractValue {
  override def nodePatternsInternal(): Seq[TreePattern] = Seq(EXTRACT_ARRAY_SUBFIELDS)
}

trait ShimGetArrayItem extends ExtractValue {
  override def nodePatternsInternal(): Seq[TreePattern] = Seq(GET_ARRAY_ITEM)
}

trait ShimGetStructField extends ExtractValue {
  override def nodePatternsInternal(): Seq[TreePattern] = Seq(GET_STRUCT_FIELD)
}

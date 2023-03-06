/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

/*** spark-rapids-shim-json-lines
{"spark": "311"}
{"spark": "312"}
{"spark": "312db"}
{"spark": "313"}
{"spark": "314"}
spark-rapids-shim-json-lines ***/
package org.apache.spark.sql.types.shims

import java.time.ZoneId

import org.apache.spark.sql.types.DataType

object PartitionValueCastShims {
  // AnyTimestamp, TimestampNTZTtpe and AnsiIntervalType types are not defined before Spark 3.2.0
  // return false between 311 until 320
  def isSupportedType(dt: DataType): Boolean = false

  def castTo(desiredType: DataType, value: String, zoneId: ZoneId): Any = {
    throw new IllegalArgumentException(s"Unexpected type $desiredType")
  }
}
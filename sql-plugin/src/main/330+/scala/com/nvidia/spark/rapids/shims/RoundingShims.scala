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

import ai.rapids.cudf.DType
import com.nvidia.spark.rapids._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution._
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.types.DecimalType

trait RoundingShims extends SparkShims {
  def roundingExprs: Map[Class[_ <: Expression], ExprRule[_ <: Expression]] = Seq(
    GpuOverrides.expr[RoundCeil](
      "Computes the ceiling of the given expression to d decimal places",
      ExprChecks.binaryProject(
        TypeSig.gpuNumeric, TypeSig.cpuNumeric,
        ("value", TypeSig.gpuNumeric +
            TypeSig.psNote(TypeEnum.FLOAT, "result may round slightly differently") +
            TypeSig.psNote(TypeEnum.DOUBLE, "result may round slightly differently"),
            TypeSig.cpuNumeric),
        ("scale", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT))),
      (ceil, conf, p, r) => new BinaryExprMeta[RoundCeil](ceil, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          ceil.child.dataType match {
            case dt: DecimalType =>
              val precision = GpuFloorCeil.unboundedOutputPrecision(dt)
              if (precision > DType.DECIMAL128_MAX_PRECISION) {
                willNotWorkOnGpu(s"output precision $precision would require overflow " +
                    s"checks, which are not supported yet")
              }
            case _ => // NOOP
          }
          GpuOverrides.extractLit(ceil.scale).foreach { scale =>
            if (scale.value != null &&
                scale.value.asInstanceOf[Integer] != 0) {
              willNotWorkOnGpu("Scale other than 0 is not supported")
            }
          }
        }

        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          // use Spark `RoundCeil.dataType` to keep consistent between Spark versions.
          GpuCeil(lhs, ceil.dataType)
        }
      }),
    GpuOverrides.expr[RoundFloor](
      "Computes the floor of the given expression to d decimal places",
      ExprChecks.binaryProject(
        TypeSig.gpuNumeric, TypeSig.cpuNumeric,
        ("value", TypeSig.gpuNumeric +
            TypeSig.psNote(TypeEnum.FLOAT, "result may round slightly differently") +
            TypeSig.psNote(TypeEnum.DOUBLE, "result may round slightly differently"),
            TypeSig.cpuNumeric),
        ("scale", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT))),
      (floor, conf, p, r) => new BinaryExprMeta[RoundFloor](floor, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          floor.child.dataType match {
            case dt: DecimalType =>
              val precision = GpuFloorCeil.unboundedOutputPrecision(dt)
              if (precision > DType.DECIMAL128_MAX_PRECISION) {
                willNotWorkOnGpu(s"output precision $precision would require overflow " +
                    s"checks, which are not supported yet")
              }
            case _ => // NOOP
          }
          GpuOverrides.extractLit(floor.scale).foreach { scale =>
            if (scale.value != null &&
                scale.value.asInstanceOf[Integer] != 0) {
              willNotWorkOnGpu("Scale other than 0 is not supported")
            }
          }
        }

        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          // use Spark `RoundFloor.dataType` to keep consistent between Spark versions.
          GpuFloor(lhs, floor.dataType)
        }
      }), 
      GpuOverrides.expr[TimeAdd](
        "Adds interval to timestamp",
        ExprChecks.binaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
          ("start", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
          // interval support DAYTIME column or CALENDAR literal
          ("interval", TypeSig.DAYTIME + TypeSig.lit(TypeEnum.CALENDAR)
              .withPsNote(TypeEnum.CALENDAR, "month intervals are not supported"),
              TypeSig.DAYTIME + TypeSig.CALENDAR)),
        (timeAdd, conf, p, r) => new BinaryExprMeta[TimeAdd](timeAdd, conf, p, r) {
          override def tagExprForGpu(): Unit = {
            GpuOverrides.extractLit(timeAdd.interval).foreach { lit =>
              lit.dataType match {
                case CalendarIntervalType =>
                  val intvl = lit.value.asInstanceOf[CalendarInterval]
                  if (intvl.months != 0) {
                    willNotWorkOnGpu("interval months isn't supported")
                  }
                case _: DayTimeIntervalType => // Supported
              }
            }
          }

          override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
            GpuTimeAdd(lhs, rhs)
        }),
        GpuOverrides.expr[Abs](
          "Absolute value",
          ExprChecks.unaryProjectAndAstInputMatchesOutput(
            TypeSig.implicitCastsAstTypes,
            TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
            TypeSig.cpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes),
          (a, conf, p, r) => new UnaryAstExprMeta[Abs](a, conf, p, r) {
            val ansiEnabled = SQLConf.get.ansiEnabled

            override def tagSelfForAst(): Unit = {
              if (ansiEnabled && GpuAnsi.needBasicOpOverflowCheck(a.dataType)) {
                willNotWorkInAst("AST unary minus does not support ANSI mode.")
              }
            }

            // ANSI support for ABS was added in 3.2.0 SPARK-33275
            override def convertToGpu(child: Expression): GpuExpression = GpuAbs(child, ansiEnabled)
          }),
      GpuOverrides.expr[MultiplyYMInterval](
        "Year-month interval * number",
        ExprChecks.binaryProject(
          TypeSig.YEARMONTH,
          TypeSig.YEARMONTH,
          ("lhs", TypeSig.YEARMONTH, TypeSig.YEARMONTH),
          ("rhs", TypeSig.gpuNumeric - TypeSig.DECIMAL_128, TypeSig.gpuNumeric)),
        (a, conf, p, r) => new BinaryExprMeta[MultiplyYMInterval](a, conf, p, r) {
          override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
            GpuMultiplyYMInterval(lhs, rhs)
        }),
      GpuOverrides.expr[MultiplyDTInterval](
        "Day-time interval * number",
        ExprChecks.binaryProject(
          TypeSig.DAYTIME,
          TypeSig.DAYTIME,
          ("lhs", TypeSig.DAYTIME, TypeSig.DAYTIME),
          ("rhs", TypeSig.gpuNumeric - TypeSig.DECIMAL_128, TypeSig.gpuNumeric)),
        (a, conf, p, r) => new BinaryExprMeta[MultiplyDTInterval](a, conf, p, r) {
          override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
            GpuMultiplyDTInterval(lhs, rhs)
        }),
      GpuOverrides.expr[DivideYMInterval](
        "Year-month interval * operator",
        ExprChecks.binaryProject(
          TypeSig.YEARMONTH,
          TypeSig.YEARMONTH,
          ("lhs", TypeSig.YEARMONTH, TypeSig.YEARMONTH),
          ("rhs", TypeSig.gpuNumeric - TypeSig.DECIMAL_128, TypeSig.gpuNumeric)),
        (a, conf, p, r) => new BinaryExprMeta[DivideYMInterval](a, conf, p, r) {
          override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
            GpuDivideYMInterval(lhs, rhs)
        }),
      GpuOverrides.expr[DivideDTInterval](
        "Day-time interval * operator",
        ExprChecks.binaryProject(
          TypeSig.DAYTIME,
          TypeSig.DAYTIME,
          ("lhs", TypeSig.DAYTIME, TypeSig.DAYTIME),
          ("rhs", TypeSig.gpuNumeric - TypeSig.DECIMAL_128, TypeSig.gpuNumeric)),
        (a, conf, p, r) => new BinaryExprMeta[DivideDTInterval](a, conf, p, r) {
          override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
            GpuDivideDTInterval(lhs, rhs)
        })
  ).map(r => (r.getClassFor.asSubclass(classOf[Expression]), r)).toMap
}

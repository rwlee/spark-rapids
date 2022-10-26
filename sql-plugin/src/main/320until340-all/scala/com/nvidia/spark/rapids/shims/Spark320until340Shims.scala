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

import org.apache.spark.sql.catalyst.expressions.{AnsiCast, Expression}
import org.apache.spark.sql.catalyst.expressions.{CastBase, CheckOverflow, Divide, Literal, Multiply, PromotePrecision}
import org.apache.spark.sql.rapids.{GpuDecimalDivide, GpuDecimalMultiply}
import org.apache.spark.sql.types.{Decimal, DecimalType}
import org.apache.spark.sql.internal.SQLConf
import ai.rapids.cudf.{DType}

trait Spark320until340Shims extends SparkShims {

  override def ansiCastRule: ExprRule[ _ <: Expression] = {
    GpuOverrides.expr[AnsiCast](
      "Convert a column of one type of data into another type",
      new CastChecks {

        import TypeSig._
        // nullChecks are the same

        override val booleanChecks: TypeSig = integral + fp + BOOLEAN + STRING
        override val sparkBooleanSig: TypeSig = cpuNumeric + BOOLEAN + STRING

        override val integralChecks: TypeSig = gpuNumeric + BOOLEAN + STRING
        override val sparkIntegralSig: TypeSig = cpuNumeric + BOOLEAN + STRING

        override val fpChecks: TypeSig = (gpuNumeric + BOOLEAN + STRING)
          .withPsNote(TypeEnum.STRING, fpToStringPsNote)
        override val sparkFpSig: TypeSig = cpuNumeric + BOOLEAN + STRING

        override val dateChecks: TypeSig = TIMESTAMP + DATE + STRING
        override val sparkDateSig: TypeSig = TIMESTAMP + DATE + STRING

        override val timestampChecks: TypeSig = TIMESTAMP + DATE + STRING
        override val sparkTimestampSig: TypeSig = TIMESTAMP + DATE + STRING

        // stringChecks are the same, but adding in PS note
        private val fourDigitYearMsg: String = "Only 4 digit year parsing is available. To " +
          s"enable parsing anyways set ${RapidsConf.HAS_EXTENDED_YEAR_VALUES} to false."
        override val stringChecks: TypeSig = gpuNumeric + BOOLEAN + STRING + BINARY +
          TypeSig.psNote(TypeEnum.DATE, fourDigitYearMsg) +
          TypeSig.psNote(TypeEnum.TIMESTAMP, fourDigitYearMsg)

        // binaryChecks are the same
        override val decimalChecks: TypeSig = gpuNumeric + STRING
        override val sparkDecimalSig: TypeSig = cpuNumeric + BOOLEAN + STRING

        // calendarChecks are the same

        override val arrayChecks: TypeSig =
          ARRAY.nested(commonCudfTypes + DECIMAL_128 + NULL + ARRAY + BINARY + STRUCT) +
            psNote(TypeEnum.ARRAY, "The array's child type must also support being cast to " +
              "the desired child type")
        override val sparkArraySig: TypeSig = ARRAY.nested(all)

        override val mapChecks: TypeSig =
          MAP.nested(commonCudfTypes + DECIMAL_128 + NULL + ARRAY + BINARY + STRUCT + MAP) +
            psNote(TypeEnum.MAP, "the map's key and value must also support being cast to the " +
              "desired child types")
        override val sparkMapSig: TypeSig = MAP.nested(all)

        override val structChecks: TypeSig =
          STRUCT.nested(commonCudfTypes + DECIMAL_128 + NULL + ARRAY + BINARY + STRUCT) +
            psNote(TypeEnum.STRUCT, "the struct's children must also support being cast to the " +
              "desired child type(s)")
        override val sparkStructSig: TypeSig = STRUCT.nested(all)

        override val udtChecks: TypeSig = none
        override val sparkUdtSig: TypeSig = UDT
      },
      (cast, conf, p, r) => new CastExprMeta[AnsiCast](cast, ansiEnabled = true, conf = conf,
        parent = p, rule = r, doFloatToIntCheck = true, stringToAnsiDate = true))
  }

  override def checkOverflowRule: ExprRule[_ <: Expression] = {

    GpuOverrides.expr[CheckOverflow](
      "CheckOverflow after arithmetic operations between DecimalType data",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.DECIMAL_128,
        TypeSig.DECIMAL_128),
      (a, conf, p, r) => new ExprMeta[CheckOverflow](a, conf, p, r) {
        private[this] def extractOrigParam(expr: BaseExprMeta[_]): BaseExprMeta[_] =
          expr.wrapped match {
            case lit: Literal if lit.dataType.isInstanceOf[DecimalType] =>
              // Lets figure out if we can make the Literal value smaller
              val (newType, value) = lit.value match {
                case null =>
                  (DecimalType(0, 0), null)
                case dec: Decimal =>
                  val stripped = Decimal(dec.toJavaBigDecimal.stripTrailingZeros())
                  val p = stripped.precision
                  val s = stripped.scale
                  val t = if (s < 0 && !SQLConf.get.allowNegativeScaleOfDecimalEnabled) {
                    // need to adjust to avoid errors about negative scale
                    DecimalType(p - s, 0)
                  } else {
                    DecimalType(p, s)
                  }
                  (t, stripped)
                case other =>
                  throw new IllegalArgumentException(s"Unexpected decimal literal value $other")
              }
              expr.asInstanceOf[LiteralExprMeta].withNewLiteral(Literal(value, newType))
            // Avoid unapply for PromotePrecision and Cast because it changes between Spark versions
            case p: PromotePrecision if p.child.isInstanceOf[CastBase] &&
                p.child.dataType.isInstanceOf[DecimalType] =>
              val c = p.child.asInstanceOf[CastBase]
              val to = c.dataType.asInstanceOf[DecimalType]
              val fromType = DecimalUtil.optionallyAsDecimalType(c.child.dataType)
              fromType match {
                case Some(from) =>
                  val minScale = math.min(from.scale, to.scale)
                  val fromWhole = from.precision - from.scale
                  val toWhole = to.precision - to.scale
                  val minWhole = if (to.scale < from.scale) {
                    // If the scale is getting smaller in the worst case we need an
                    // extra whole part to handle rounding up.
                    math.min(fromWhole + 1, toWhole)
                  } else {
                    math.min(fromWhole, toWhole)
                  }
                  val newToType = DecimalType(minWhole + minScale, minScale)
                  if (newToType == from) {
                    // We can remove the cast totally
                    val castExpr = expr.childExprs.head
                    castExpr.childExprs.head
                  } else if (newToType == to) {
                    // The cast is already ideal
                    expr
                  } else {
                    val castExpr = expr.childExprs.head.asInstanceOf[CastExprMeta[_]]
                    castExpr.withToTypeOverride(newToType)
                  }
                case _ =>
                  expr
              }
            case _ => expr
          }
        private[this] lazy val binExpr = childExprs.head
        private[this] lazy val lhs = extractOrigParam(binExpr.childExprs.head)
        private[this] lazy val rhs = extractOrigParam(binExpr.childExprs(1))
        private[this] lazy val lhsDecimalType =
          DecimalUtil.asDecimalType(lhs.wrapped.asInstanceOf[Expression].dataType)
        private[this] lazy val rhsDecimalType =
          DecimalUtil.asDecimalType(rhs.wrapped.asInstanceOf[Expression].dataType)

        override def convertToGpu(): GpuExpression = {
          // Prior to Spark 3.4.0
          // Division and Multiplication of Decimal types is a little odd. Spark will cast the
          // inputs to a common wider value where the scale is the max of the two input scales,
          // and the precision is max of the two input non-scale portions + the new scale. Then it
          // will do the divide or multiply as a BigDecimal value but lie about the return type.
          // Finally here in CheckOverflow it will reset the scale and check the precision so that
          // Spark knows it fits in the final desired result.
          // Here we try to strip out the extra casts, etc to get to as close to the original
          // query as possible. This lets us then calculate what CUDF needs to get the correct
          // answer, which in some cases is a lot smaller.

          a.child match {
            case _: Divide =>
              // GpuDecimalDivide includes the overflow check in it.
              GpuDecimalDivide(lhs.convertToGpu(), rhs.convertToGpu(), wrapped.dataType)
            case _: Multiply =>
              // GpuDecimal*Multiply includes the overflow check in it
              val intermediatePrecision =
                GpuDecimalMultiply.nonRoundedIntermediatePrecision(lhsDecimalType,
                  rhsDecimalType, a.dataType)
              GpuDecimalMultiply(lhs.convertToGpu(), rhs.convertToGpu(), wrapped.dataType,
                useLongMultiply = intermediatePrecision > DType.DECIMAL128_MAX_PRECISION)
            case _ =>
              GpuCheckOverflow(childExprs.head.convertToGpu(),
                wrapped.dataType, wrapped.nullOnOverflow)
          }
        }
      })
  }

  override def promotePrecisionRule: ExprRule[_ <: Expression] = {
    GpuOverrides.expr[PromotePrecision](
      "PromotePrecision before arithmetic operations between DecimalType data",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.DECIMAL_128,
        TypeSig.DECIMAL_128),
      (a, conf, p, r) => new UnaryExprMeta[PromotePrecision](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuPromotePrecision(child)
      })
  }

  override def ignoreTimeZoneCastBase(e: Expression): Expression = e match {
    case c: CastBase if c.timeZoneId.nonEmpty && !c.needsTimeZone =>
      c.withTimeZone(null)
    case c: GpuCast if c.timeZoneId.nonEmpty && !c.needsTimeZone =>
      c.withTimeZone(null)
    case _ => e
  }
}

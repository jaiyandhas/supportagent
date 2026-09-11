import React from "react";
import type { EscalationDecision, ReliabilityPoint } from "../types";

interface TrustPanelProps {
  escalation: EscalationDecision;
  reliabilityPoints?: ReliabilityPoint[];
  trustEce?: number;
  baselineEce?: number;
}

export const TrustPanel: React.FC<TrustPanelProps> = ({
  escalation,
  reliabilityPoints = [],
  trustEce = 0.369,
  baselineEce = 0.483,
}) => {
  const isAutoHandled = escalation.status === "auto_handled";
  const confidencePct = Math.round(escalation.calibrated_confidence * 100);

  // SVG dimensions for the inline calibration curve
  const width = 280;
  const height = 150;
  const padding = 26;
  const plotW = width - padding * 2;
  const plotH = height - padding * 2;

  // Filter non-empty bins for reliable plotting
  const validPoints = reliabilityPoints.filter((p) => p.sample_count > 0);

  return (
    <div className="mt-6 p-5 bg-white border border-[#D2D2D7] rounded-xl">
      <div className="flex items-center justify-between pb-4 border-b border-[#D2D2D7]">
        <div>
          <h3 className="text-[17px] font-medium text-[#1D1D1F]">
            Trust & calibration gate
          </h3>
          <p className="text-[13px] text-[#6E6E73] mt-0.5">
            ECA-RAG adaptive-k agreement with Platt scaling calibration
          </p>
        </div>

        <div className="text-right">
          <div className="text-[32px] font-bold tracking-tight text-[#0071E3] leading-none">
            {confidencePct}%
          </div>
          <div className="text-[11px] text-[#6E6E73] mt-1">
            Calibrated confidence
          </div>
        </div>
      </div>

      {/* Stated Reason */}
      <div className="py-4">
        <div className="text-[13px] text-[#6E6E73] mb-1">
          System routing rationale
        </div>
        <p className="text-[14px] text-[#1D1D1F] leading-relaxed">
          {escalation.stated_reason}
        </p>
      </div>

      {/* Interactive Calibration Curve / Reliability Diagram */}
      <div className="pt-4 border-t border-[#D2D2D7]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[13px] font-medium text-[#1D1D1F]">
            Calibration curve (ECE {trustEce.toFixed(3)})
          </span>
          <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#F5F5F7] text-[#6E6E73]">
            Simple baseline ECE: {baselineEce.toFixed(3)}
          </span>
        </div>

        <div className="flex items-center justify-center my-1">
          <svg width={width} height={height} className="overflow-visible">
            {/* Background Grid & Axes */}
            <rect
              x={padding}
              y={padding}
              width={plotW}
              height={plotH}
              fill="#FAFAFA"
              stroke="#E5E5EA"
              strokeWidth="1"
            />

            {/* Ideal diagonal calibration line y = x */}
            <line
              x1={padding}
              y1={padding + plotH}
              x2={padding + plotW}
              y2={padding}
              stroke="#D2D2D7"
              strokeWidth="1.5"
              strokeDasharray="3 3"
            />

            {/* Operating Threshold line (tau = 0.62) */}
            <line
              x1={padding + 0.62 * plotW}
              y1={padding}
              x2={padding + 0.62 * plotW}
              y2={padding + plotH}
              stroke="#0071E3"
              strokeWidth="1"
              strokeDasharray="2 2"
            />

            {/* Plot Reliability Bins */}
            {validPoints.map((pt, i) => {
              const cx = padding + pt.mean_confidence * plotW;
              const cy = padding + plotH - pt.empirical_accuracy * plotH;
              return (
                <g key={i}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={Math.min(6, Math.max(3, pt.sample_count * 0.4))}
                    fill="#0071E3"
                    opacity={0.85}
                  />
                </g>
              );
            })}

            {/* Current query confidence marker */}
            <circle
              cx={padding + escalation.calibrated_confidence * plotW}
              cy={
                padding +
                plotH -
                (isAutoHandled ? escalation.calibrated_confidence : 0.05) * plotH
              }
              r={5}
              fill={isAutoHandled ? "#2E7D5B" : "#B25E09"}
              stroke="#FFFFFF"
              strokeWidth="2"
            />

            {/* Axis labels */}
            <text
              x={padding}
              y={height - 6}
              fontSize="10"
              fill="#86868B"
              textAnchor="start"
            >
              0.0
            </text>
            <text
              x={padding + plotW / 2}
              y={height - 6}
              fontSize="10"
              fill="#86868B"
              textAnchor="middle"
            >
              Predicted confidence
            </text>
            <text
              x={padding + plotW}
              y={height - 6}
              fontSize="10"
              fill="#86868B"
              textAnchor="end"
            >
              1.0
            </text>
          </svg>
        </div>

        <div className="flex items-center justify-between text-[11px] text-[#86868B] mt-1 px-1">
          <span>Operating threshold: 0.62</span>
          <span>Adaptive evidence: k = {escalation.adaptive_k}</span>
          <span>Consensus score: {escalation.raw_agreement_score.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

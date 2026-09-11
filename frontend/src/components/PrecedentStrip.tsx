import React from "react";
import type { PrecedentCandidate } from "../types";

interface PrecedentStripProps {
  precedents: PrecedentCandidate[];
}

export const PrecedentStrip: React.FC<PrecedentStripProps> = ({ precedents }) => {
  if (!precedents || precedents.length === 0) {
    return (
      <div className="p-4 border border-[#D2D2D7] rounded-lg bg-white text-[13px] text-[#6E6E73]">
        No historical precedents cited for this inquiry.
      </div>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[15px] font-medium text-[#1D1D1F]">
          Outcome-verified precedent grounding
        </h3>
        <span className="text-[13px] text-[#6E6E73]">
          {precedents.length} precedent{precedents.length > 1 ? "s" : ""} retrieved & reranked
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {precedents.map((prec) => {
          const isHighRes = prec.resolution_score >= 0.70;
          const isLowRes = prec.resolution_score < 0.40;

          return (
            <div
              key={prec.precedent_id}
              className="p-3.5 bg-white border border-[#D2D2D7] rounded-lg flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[12px] text-[#6E6E73]">
                    {prec.precedent_id}
                  </span>
                  <span
                    className={`text-[12px] px-2 py-0.5 rounded-full ${
                      isHighRes
                        ? "bg-[#EAF3EE] text-[#2E7D5B]"
                        : isLowRes
                        ? "bg-[#FBF0E1] text-[#B25E09]"
                        : "bg-[#F5F5F7] text-[#6E6E73]"
                    }`}
                  >
                    Resolution {(prec.resolution_score * 100).toFixed(0)}%
                  </span>
                </div>

                <p className="text-[13px] text-[#1D1D1F] leading-relaxed mb-2">
                  "{prec.support_reply}"
                </p>
              </div>

              <div className="pt-2 border-t border-[#F5F5F7] flex items-center justify-between text-[11px] text-[#86868B]">
                <span>Similarity {(prec.semantic_similarity * 100).toFixed(0)}%</span>
                <span>Reranked {(prec.reranked_score * 100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

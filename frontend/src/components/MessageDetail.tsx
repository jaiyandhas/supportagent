import React from "react";
import type { QueueMessage, ReliabilityPoint } from "../types";
import { PrecedentStrip } from "./PrecedentStrip";
import { TrustPanel } from "./TrustPanel";

interface MessageDetailProps {
  message: QueueMessage | null;
  reliabilityPoints: ReliabilityPoint[];
  trustEce: number;
  baselineEce: number;
  activeProvider?: string;
}

export const MessageDetail: React.FC<MessageDetailProps> = ({
  message,
  reliabilityPoints,
  trustEce,
  baselineEce,
  activeProvider = "Local Engine",
}) => {
  if (!message) {
    return (
      <div className="h-full flex items-center justify-center p-8 text-[#86868B] text-[15px]">
        Select a customer message from the queue to audit its decision and precedent grounding.
      </div>
    );
  }

  const { prediction } = message;
  const isAutoHandled = prediction.escalation.status === "auto_handled";
  const isLive = prediction.generation_mode === "live_llm";

  return (
    <div className="h-full overflow-y-auto p-8 crossfade-pane">
      {/* Header bar */}
      <div className="flex items-center justify-between pb-6 border-b border-[#D2D2D7]">
        <div className="flex items-center space-x-3">
          <span
            className={`text-[13px] px-3 py-1 rounded-full font-medium ${
              isAutoHandled
                ? "bg-[#EAF3EE] text-[#2E7D5B]"
                : "bg-[#FBF0E1] text-[#B25E09]"
            }`}
          >
            {isAutoHandled ? "Auto-handled" : "Escalated to human"}
          </span>

          <span className="text-[13px] px-2.5 py-0.5 rounded-full bg-[#F5F5F7] text-[#6E6E73]">
            {prediction.intent.coarse_intent.replace("_", " ")}
          </span>

          <span className="text-[13px] text-[#86868B]">
            {prediction.intent.sub_intent.replace(/_/g, " ")}
          </span>
        </div>

        <div className="text-[12px] text-[#86868B]">
          {message.id}
        </div>
      </div>

      {/* Customer Message */}
      <div className="mt-6">
        <div className="text-[13px] text-[#6E6E73] mb-2">
          Customer message (@AmazonHelp)
        </div>
        <div className="p-4 bg-white border border-[#D2D2D7] rounded-xl text-[15px] text-[#1D1D1F] leading-relaxed">
          "{message.customer_query}"
        </div>
      </div>

      {/* Drafted Reply */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[13px] text-[#6E6E73]">
            Drafted response
          </span>
          <span className="text-[12px] text-[#6E6E73] flex items-center gap-1">
            {isLive ? (
              <span className="text-[#0071E3] font-medium">⚡ Live LLM ({activeProvider})</span>
            ) : (
              <span className="text-[#86868B]">🔒 Deterministic Repro Engine</span>
            )}
          </span>
        </div>

        <div className="p-4 bg-white border border-[#D2D2D7] rounded-xl text-[15px] text-[#1D1D1F] leading-relaxed">
          {prediction.drafted_reply}
        </div>
      </div>

      {/* Precedent Strip (Phase 2B grounding evidence) */}
      <PrecedentStrip precedents={prediction.precedents} />

      {/* Trust & Calibration Panel (Phase 2C & Phase 6) */}
      <TrustPanel
        escalation={prediction.escalation}
        reliabilityPoints={reliabilityPoints}
        trustEce={trustEce}
        baselineEce={baselineEce}
      />
    </div>
  );
};

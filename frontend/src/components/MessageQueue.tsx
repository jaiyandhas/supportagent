import React, { useState } from "react";
import type { QueueMessage } from "../types";

interface MessageQueueProps {
  messages: QueueMessage[];
  selectedId: string | null;
  onSelect: (msg: QueueMessage) => void;
  onTestQuery: (query: string) => void;
  isPredicting: boolean;
}

export const MessageQueue: React.FC<MessageQueueProps> = ({
  messages,
  selectedId,
  onSelect,
  onTestQuery,
  isPredicting,
}) => {
  const [filter, setFilter] = useState<"all" | "auto" | "escalated" | "cold">("all");
  const [search, setSearch] = useState("");
  const [testInput, setTestInput] = useState("");

  const filteredMessages = messages.filter((m) => {
    if (filter === "auto" && m.prediction.escalation.status !== "auto_handled") return false;
    if (filter === "escalated" && m.prediction.escalation.status !== "escalated") return false;
    if (filter === "cold" && !m.is_cold_case) return false;
    if (search.trim()) {
      const q = search.toLowerCase();
      return (
        m.customer_query.toLowerCase().includes(q) ||
        m.prediction.intent.coarse_intent.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!testInput.trim() || isPredicting) return;
    onTestQuery(testInput.trim());
    setTestInput("");
  };

  return (
    <div className="h-full flex flex-col border-r border-[#D2D2D7] bg-[#FAFAFA]">
      {/* Top filter header */}
      <div className="p-4 border-b border-[#D2D2D7]">
        <div className="flex items-center justify-between mb-3">
          <span className="text-[17px] font-medium text-[#1D1D1F]">
            Auditable queue
          </span>
          <span className="text-[13px] text-[#6E6E73]">
            {filteredMessages.length} inquiries
          </span>
        </div>

        {/* Filter Pills */}
        <div className="flex space-x-1 mb-3">
          {(
            [
              { id: "all", label: "All" },
              { id: "auto", label: "Auto-handled" },
              { id: "escalated", label: "Escalated" },
              { id: "cold", label: "Cold cases" },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              onClick={() => setFilter(item.id)}
              className={`text-[12px] px-2.5 py-1 rounded-full transition-colors ${
                filter === item.id
                  ? "bg-[#1D1D1F] text-white font-medium"
                  : "bg-white text-[#6E6E73] border border-[#D2D2D7] hover:bg-[#F5F5F7]"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Search input */}
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter messages or intents..."
          className="w-full px-3 py-1.5 text-[13px] bg-white border border-[#D2D2D7] rounded-lg focus:outline-none focus:border-[#0071E3]"
        />
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto divide-y divide-[#D2D2D7]">
        {filteredMessages.map((msg) => {
          const isSelected = selectedId === msg.id;
          const isAuto = msg.prediction.escalation.status === "auto_handled";

          return (
            <div
              key={msg.id}
              onClick={() => onSelect(msg)}
              className={`p-4 cursor-pointer transition-colors ${
                isSelected
                  ? "bg-white border-l-3 border-[#0071E3]"
                  : "hover:bg-[#F5F5F7]"
              }`}
            >
              <p className="text-[14px] text-[#1D1D1F] line-clamp-2 leading-snug mb-2.5">
                {msg.customer_query}
              </p>

              <div className="flex items-center space-x-2">
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#F5F5F7] text-[#6E6E73]">
                  {msg.prediction.intent.coarse_intent.replace("_", " ")}
                </span>

                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full ${
                    isAuto
                      ? "bg-[#EAF3EE] text-[#2E7D5B]"
                      : "bg-[#FBF0E1] text-[#B25E09]"
                  }`}
                >
                  {isAuto ? "Auto-handled" : "Escalated"}
                </span>

                {msg.is_cold_case && (
                  <span className="text-[11px] px-1.5 py-0.5 rounded bg-neutral-200 text-neutral-700">
                    Cold case
                  </span>
                )}
              </div>
            </div>
          );
        })}

        {filteredMessages.length === 0 && (
          <div className="p-8 text-center text-[13px] text-[#86868B]">
            No inquiries match the active filter.
          </div>
        )}
      </div>

      {/* Real-time Interactive Test Form */}
      <div className="p-3 border-t border-[#D2D2D7] bg-white">
        <form onSubmit={handleCustomSubmit} className="flex gap-2">
          <input
            type="text"
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
            placeholder="Type custom test tweet..."
            className="flex-1 px-3 py-1.5 text-[13px] bg-[#FAFAFA] border border-[#D2D2D7] rounded-lg focus:outline-none focus:border-[#0071E3]"
          />
          <button
            type="submit"
            disabled={isPredicting || !testInput.trim()}
            className="px-3.5 py-1.5 text-[13px] bg-[#0071E3] text-white rounded-lg font-medium hover:bg-[#0077ED] disabled:opacity-50"
          >
            {isPredicting ? "..." : "Evaluate"}
          </button>
        </form>
      </div>
    </div>
  );
};

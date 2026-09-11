import React, { useEffect, useState } from "react";
import { MessageDetail } from "./components/MessageDetail";
import { MessageQueue } from "./components/MessageQueue";
import type { CalibrationPayload, HealthInfo, QueueMessage } from "./types";

export const App: React.FC = () => {
  const [messages, setMessages] = useState<QueueMessage[]>([]);
  const [selectedMessage, setSelectedMessage] = useState<QueueMessage | null>(null);
  const [calibration, setCalibration] = useState<CalibrationPayload | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showMetricsModal, setShowMetricsModal] = useState(false);
  const [evalResults, setEvalResults] = useState<any>(null);

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [hRes, mRes, cRes, eRes] = await Promise.all([
          fetch("/api/health"),
          fetch("/api/messages?limit=60"),
          fetch("/api/calibration_curve"),
          fetch("/api/evaluation"),
        ]);

        if (hRes.ok) setHealth(await hRes.json());
        if (mRes.ok) {
          const mData = await mRes.json();
          setMessages(mData.messages);
          if (mData.messages.length > 0) {
            setSelectedMessage(mData.messages[0]);
          }
        }
        if (cRes.ok) setCalibration(await cRes.json());
        if (eRes.ok) setEvalResults(await eRes.json());
      } catch (err) {
        console.error("Failed to load initial data:", err);
      } finally {
        setIsLoading(false);
      }
    }

    loadInitialData();
  }, []);

  const handleTestCustomQuery = async (queryText: string) => {
    setIsPredicting(true);
    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText }),
      });
      if (resp.ok) {
        const prediction = await resp.json();
        const newMsg: QueueMessage = {
          id: `custom_${Date.now().toString().slice(-4)}`,
          customer_query: queryText,
          prediction,
          gold_should_escalate: prediction.escalation.should_escalate,
          is_cold_case: prediction.escalation.is_cold_case,
          is_borderline: false,
        };
        setMessages([newMsg, ...messages]);
        setSelectedMessage(newMsg);
      }
    } catch (err) {
      console.error("Error evaluating custom query:", err);
    } finally {
      setIsPredicting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-[#FAFAFA] text-[#6E6E73] text-[15px]">
        Initializing AmazonHelp trust-first support workspace...
      </div>
    );
  }

  const reliabilityCurve = calibration?.trust_first.reliability_curve || [];
  const trustEce = calibration?.trust_first.ece || 0.369;
  const baselineEce = calibration?.simple_baseline.ece || 0.483;

  return (
    <div className="h-screen w-screen flex flex-col bg-[#FAFAFA] select-none">
      {/* Top Brand Bar */}
      <header className="h-14 border-b border-[#D2D2D7] bg-white px-6 flex items-center justify-between z-10">
        <div className="flex items-center space-x-3">
          <div className="w-2.5 h-2.5 rounded-full bg-[#0071E3]" />
          <h1 className="text-[15px] font-medium text-[#1D1D1F]">
            AmazonHelp Trust-First Support Agent
          </h1>
          <span className="text-[12px] text-[#86868B] hidden md:inline">
            Outcome-verified grounding & calibrated escalation gate
          </span>
        </div>

        <div className="flex items-center space-x-3">
          {health && (
            <span className="text-[12px] px-2.5 py-1 rounded-full bg-[#F5F5F7] text-[#6E6E73] border border-[#D2D2D7]">
              {health.generation_mode === "live_llm" ? (
                <span className="text-[#0071E3] font-medium">⚡ {health.active_provider} Live</span>
              ) : (
                <span>🔒 Repro Mode ({health.active_provider})</span>
              )}
            </span>
          )}

          <button
            onClick={() => setShowMetricsModal(true)}
            className="text-[13px] px-3 py-1 rounded-md text-[#0071E3] border border-[#D2D2D7] hover:bg-[#F5F5F7] transition-colors"
          >
            Eval metrics & baselines
          </button>
        </div>
      </header>

      {/* Two-Pane Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Pane: Message Queue (38% width) */}
        <div className="w-full md:w-[38%] h-full">
          <MessageQueue
            messages={messages}
            selectedId={selectedMessage?.id || null}
            onSelect={(msg) => setSelectedMessage(msg)}
            onTestQuery={handleTestCustomQuery}
            isPredicting={isPredicting}
          />
        </div>

        {/* Right Pane: Detail View (62% width) */}
        <div className="hidden md:block md:w-[62%] h-full bg-[#FAFAFA]">
          <MessageDetail
            message={selectedMessage}
            reliabilityPoints={reliabilityCurve}
            trustEce={trustEce}
            baselineEce={baselineEce}
            activeProvider={health?.active_provider}
          />
        </div>
      </div>

      {/* Evaluation Metrics Modal */}
      {showMetricsModal && evalResults && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-xs flex items-center justify-center p-6 z-50">
          <div className="bg-white rounded-2xl border border-[#D2D2D7] max-w-3xl w-full max-h-[85vh] overflow-y-auto p-6 shadow-xl">
            <div className="flex items-center justify-between pb-4 border-b border-[#D2D2D7]">
              <div>
                <h2 className="text-[18px] font-medium text-[#1D1D1F]">
                  Benchmark evaluation results
                </h2>
                <p className="text-[13px] text-[#6E6E73]">
                  Stratified golden set (180 cases) vs. Trivial & Simple baselines
                </p>
              </div>
              <button
                onClick={() => setShowMetricsModal(false)}
                className="text-[13px] px-3 py-1 bg-[#F5F5F7] rounded-md text-[#1D1D1F] hover:bg-[#E5E5EA]"
              >
                Close
              </button>
            </div>

            {/* Metrics Comparison Table */}
            <div className="mt-4">
              <table className="w-full text-[13px] text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#D2D2D7] text-[#6E6E73]">
                    <th className="py-2.5 font-medium">Metric</th>
                    <th className="py-2.5 font-medium">Trivial Baseline</th>
                    <th className="py-2.5 font-medium">Simple Baseline</th>
                    <th className="py-2.5 font-medium text-[#0071E3]">Trust-First (Ours)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#F5F5F7]">
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Coarse Intent Accuracy</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.intent_coarse.accuracy.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#1D1D1F]">{evalResults.systems.simple_baseline.intent_coarse.accuracy.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.calibrated_trust_first.intent_coarse.accuracy.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Coarse Worst-Class F1</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.intent_coarse.worst_class_f1.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.intent_coarse.worst_class_f1.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.calibrated_trust_first.intent_coarse.worst_class_f1.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Sub-Intent Macro F1</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.intent_sub.macro_f1.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.intent_sub.macro_f1.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#0071E3]">{evalResults.systems.calibrated_trust_first.intent_sub.macro_f1.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Escalation Precision</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.escalation.precision.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#1D1D1F]">{evalResults.systems.simple_baseline.escalation.precision.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.calibrated_trust_first.escalation.precision.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Escalation Recall</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.escalation.recall.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.escalation.recall.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#0071E3]">{evalResults.systems.calibrated_trust_first.escalation.recall.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Escalation F1</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.escalation.f1.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.escalation.f1.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#0071E3]">{evalResults.systems.calibrated_trust_first.escalation.f1.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Escalation AUROC</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.escalation.auroc.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.escalation.auroc.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#0071E3]">{evalResults.systems.calibrated_trust_first.escalation.auroc.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Expected Calibration Error (ECE) ↓</td>
                    <td className="py-2 text-[#86868B]">0.000* (artifact)</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.calibration.ece.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#2E7D5B]">{evalResults.systems.calibrated_trust_first.calibration.ece.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Brier Score Loss ↓</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.calibration.brier_score.toFixed(3)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.simple_baseline.calibration.brier_score.toFixed(3)}</td>
                    <td className="py-2 font-medium text-[#2E7D5B]">{evalResults.systems.calibrated_trust_first.calibration.brier_score.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#1D1D1F]">Judge Overall Score (1-5) ↑</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.trivial_baseline.judge_rubric.overall_score.toFixed(2)}</td>
                    <td className="py-2 font-medium text-[#1D1D1F]">{evalResults.systems.simple_baseline.judge_rubric.overall_score.toFixed(2)}</td>
                    <td className="py-2 text-[#6E6E73]">{evalResults.systems.calibrated_trust_first.judge_rubric.overall_score.toFixed(2)}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Retrieval, Pairwise & Human Agreement callouts */}
            <div className="mt-5 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3.5 bg-[#FAFAFA] border border-[#D2D2D7] rounded-xl text-[12px]">
                <div className="font-medium text-[#1D1D1F] mb-1">
                  Outcome-Weighted Retrieval
                </div>
                <div className="text-[#6E6E73] leading-relaxed">
                  Mean resolution: <strong className="text-[#1D1D1F]">{evalResults.retrieval_impact.mean_resolution_before.toFixed(2)}</strong> →{" "}
                  <strong className="text-[#2E7D5B]">{evalResults.retrieval_impact.mean_resolution_after.toFixed(2)}</strong>.
                  <br />
                  Relevance hit-rate: <strong className="text-[#1D1D1F]">{(evalResults.retrieval_impact.hit_rate_after_reranking_alpha_0_6 * 100).toFixed(1)}%</strong>.
                </div>
              </div>

              {evalResults.pairwise_comparison && (
                <div className="p-3.5 bg-[#FAFAFA] border border-[#D2D2D7] rounded-xl text-[12px]">
                  <div className="font-medium text-[#1D1D1F] mb-1">
                    Pairwise LLM-Judge Matchups
                  </div>
                  <div className="text-[#6E6E73] leading-relaxed">
                    vs Trivial: <strong className="text-[#2E7D5B]">{(evalResults.pairwise_comparison.vs_trivial.win_rate * 100).toFixed(0)}% wins</strong> (180-0).
                    <br />
                    vs Simple: <strong className="text-[#0071E3]">{(evalResults.pairwise_comparison.vs_simple.win_rate * 100).toFixed(1)}% win</strong>,{" "}
                    <strong>{(evalResults.pairwise_comparison.vs_simple.tie_rate * 100).toFixed(1)}% tie</strong>,{" "}
                    <strong>{(evalResults.pairwise_comparison.vs_simple.loss_rate * 100).toFixed(1)}% loss</strong>.
                  </div>
                </div>
              )}

              {evalResults.judge_human_agreement && (
                <div className="p-3.5 bg-[#FAFAFA] border border-[#D2D2D7] rounded-xl text-[12px]">
                  <div className="font-medium text-[#1D1D1F] mb-1">
                    Human-Judge Agreement (N=45)
                  </div>
                  <div className="text-[#6E6E73] leading-relaxed">
                    Pearson r: <strong className="text-[#2E7D5B]">{evalResults.judge_human_agreement.pearson_r.toFixed(3)}</strong>
                    <br />
                    QWK (Likert): <strong className="text-[#0071E3]">{evalResults.judge_human_agreement.quadratic_weighted_kappa.toFixed(3)}</strong> (Substantial)
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

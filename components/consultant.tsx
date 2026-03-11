"use client";
import { useState, useRef, useEffect } from "react";

// ─── SYSTEM PROMPTS ──────────────────────────────────────────────────────────

const DISCOVERY_SYSTEM = `You are an expert AI Custom Integration Consultant. Your job is to deeply understand a company's operations, then recommend exactly which AI systems can be built and integrated into their business.

DISCOVERY PHASE RULES:
- Ask ONE focused question at a time
- After 3-4 exchanges you should have enough to generate a report
- Ask about: what they do, their current workflow/tools, biggest time sinks or pain points, team size
- Be conversational and warm — this is a consultative discovery call
- When you have enough info, end your message with exactly: [READY_FOR_REPORT]
- Do not add any text after [READY_FOR_REPORT]

Keep responses concise. You're on a discovery call, not writing an essay.`;

const REPORT_SYSTEM = `You are an expert AI integration consultant. Based on the discovery conversation, generate a comprehensive AI integration report.

Return ONLY valid JSON (no markdown, no backticks, no preamble) in this exact structure:
{
  "company_summary": "2-sentence summary of what this company does",
  "industry": "Industry name",
  "key_pain_points": ["pain point 1", "pain point 2", "pain point 3"],
  "integrations": [
    {
      "id": "unique_id",
      "title": "Integration title (specific, not generic)",
      "category": "one of: Customer Experience | Operations | Sales & Marketing | Data & Analytics | Internal Tools | Content & Communications",
      "tagline": "One punchy sentence on the value",
      "what_it_does": "2-3 sentences describing exactly what this AI integration does for their specific business",
      "how_we_build_it": "Specific technical approach: which AI APIs, how it connects to their existing tools, data flow",
      "tools_and_apis": ["Tool 1", "Tool 2", "Tool 3"],
      "impact": "High|Medium|Low",
      "effort": "High|Medium|Low",
      "timeline": "X-Y weeks",
      "roi_signal": "Specific measurable outcome (e.g. 40% reduction in support tickets)",
      "first_step": "The single most important first action to get started"
    }
  ],
  "quick_wins": ["specific quick win 1", "specific quick win 2"],
  "recommended_start": "Which integration to build first and exactly why",
  "total_opportunity": "High-level summary of the transformation opportunity"
}

Generate 4-6 integrations. Make them SPECIFIC to this company — not generic AI suggestions. Reference their actual workflows, tools, and pain points.`;

// ─── API CALLS ────────────────────────────────────────────────────────────────

import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY || "", 
});

type Message = { role: string; content: string };
async function streamMessage(messages: Message[], systemPrompt: string, onChunk: (text: string) => void) {
  const contents = messages.map(m => ({
    role: m.role === "user" ? "user" : "model",
    parts: [{ text: m.content }]
  }));

  const responseStream = await ai.models.generateContentStream({
    model: "gemini-3-flash-preview",
    contents: contents,
    config: { systemInstruction: systemPrompt },
  });

  let full = "";
  for await (const chunk of responseStream) {
    if (chunk.text) {
      full += chunk.text;
      onChunk(full);
    }
  }
  return full;
}

function robustJsonParse(text: string) {
  try {
    // 1. Try direct parse (standard case)
    return JSON.parse(text);
  } catch {
    console.warn("Standard JSON parse failed, attempting robust extraction...");
    
    // 2. Try to strip markdown code blocks if they exist
    const cleanText = text.replace(/```json\n?|```/g, "").trim();
    try {
      return JSON.parse(cleanText);
    } catch {
      // 3. Try to extract the first/largest JSON object/array
      const firstCurly = text.indexOf('{');
      const lastCurly = text.lastIndexOf('}');
      const firstBracket = text.indexOf('[');
      const lastBracket = text.lastIndexOf(']');
      
      let start = -1;
      let end = -1;
      
      if (firstCurly !== -1 && (firstBracket === -1 || firstCurly < firstBracket)) {
        start = firstCurly;
        end = lastCurly;
      } else if (firstBracket !== -1) {
        start = firstBracket;
        end = lastBracket;
      }

      if (start !== -1 && end !== -1 && end > start) {
        const extracted = text.substring(start, end + 1);
        try {
          return JSON.parse(extracted);
        } catch (e3) {
          console.error("Robust JSON extraction failed:", e3);
        }
      }
      
      console.error("All JSON parsing attempts failed for text:", text);
      return null;
    }
  }
}

async function generateReport(conversation: string) {
  const response = await ai.models.generateContent({
    model: "gemini-3-pro-preview",
    contents: `Here is the discovery conversation:\n\n${conversation}\n\nGenerate the AI integration report JSON now.`,
    config: {
      systemInstruction: REPORT_SYSTEM,
      responseMimeType: "application/json"
    }
  });

  const text = response.text || "";
  if (!text) return null;

  return robustJsonParse(text);
}

// ─── CONSTANTS ────────────────────────────────────────────────────────────────

const IMPACT_COLOR = { High: "#22c55e", Medium: "#f59e0b", Low: "#94a3b8" };
const EFFORT_COLOR = { High: "#ef4444", Medium: "#f59e0b", Low: "#22c55e" };
const CAT_ICONS: Record<string, string> = {
  "Customer Experience": "◎",
  "Operations": "⟳",
  "Sales & Marketing": "◈",
  "Data & Analytics": "≋",
  "Internal Tools": "⊞",
  "Content & Communications": "✦",
};

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      background: color + "18", color, border: `1px solid ${color}40`,
      fontSize: "10px", fontWeight: 700, padding: "2px 8px",
      borderRadius: "2px", letterSpacing: "0.08em", fontFamily: "Georgia, serif",
      textTransform: "uppercase",
    }}>{label}</span>
  );
}

interface IntegrationItem {
  id?: string | number;
  category?: string;
  title?: string;
  tagline?: string;
  impact?: string;
  effort?: string;
  description?: string;
  steps?: string[];
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

interface ReportData {
  company_summary?: string;
  industry?: string;
  integrations?: IntegrationItem[];
  key_pain_points?: string[];
  recommended_start?: string;
  total_opportunity?: string;
  quick_wins?: string[];
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

function IntegrationCard({ item }: { item: IntegrationItem; index?: number }) {
  const [open, setOpen] = useState(false);
  const icon = CAT_ICONS[item.category || ""] || "◈";
  return (
    <div style={{
      background: "#fff", border: "1px solid #e8e0d4",
      marginBottom: "12px", overflow: "hidden",
      boxShadow: open ? "0 8px 32px rgba(0,0,0,0.08)" : "0 2px 8px rgba(0,0,0,0.04)",
      transition: "box-shadow 0.2s",
    }}>
      <div
        onClick={() => setOpen(!open)}
        style={{
          padding: "20px 24px", cursor: "pointer",
          display: "flex", alignItems: "flex-start", gap: "16px",
          background: open ? "#faf8f5" : "#fff",
          borderBottom: open ? "1px solid #e8e0d4" : "none",
        }}
      >
        <div style={{
          width: "44px", height: "44px", background: "#1a1208",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "18px", color: "#c9a84c", flexShrink: 0,
        }}>{icon}</div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap", marginBottom: "4px" }}>
            <span style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.15em", textTransform: "uppercase", fontFamily: "Georgia, serif" }}>{item.category}</span>
          </div>
          <div style={{ fontSize: "16px", fontWeight: 700, color: "#1a1208", fontFamily: "Georgia, serif", marginBottom: "4px" }}>{item.title}</div>
          <div style={{ fontSize: "13px", color: "#6b5b3e" }}>{item.tagline}</div>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center", flexShrink: 0 }}>
          <Badge label={`↑ ${item.impact || "N/A"}`} color={(IMPACT_COLOR as Record<string, string>)[item.impact || ""] || "#94a3b8"} />
          <span style={{ fontSize: "18px", color: "#c9a84c", transition: "transform 0.2s", transform: open ? "rotate(180deg)" : "none" }}>∨</span>
        </div>
      </div>
      {open && (
        <div style={{ padding: "24px", background: "#faf8f5" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
            <div>
              <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "6px", fontFamily: "Georgia, serif" }}>What It Does</div>
              <div style={{ fontSize: "13px", color: "#3d2e1a", lineHeight: 1.7 }}>{item.what_it_does}</div>
            </div>
            <div>
              <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "6px", fontFamily: "Georgia, serif" }}>How We Build It</div>
              <div style={{ fontSize: "13px", color: "#3d2e1a", lineHeight: 1.7 }}>{item.how_we_build_it}</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", marginBottom: "20px" }}>
            {(item.tools_and_apis || []).map((t: string) => (
              <span key={t} style={{
                background: "#1a1208", color: "#c9a84c",
                fontSize: "11px", padding: "4px 10px", fontFamily: "monospace",
              }}>{t}</span>
            ))}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", marginBottom: "20px" }}>
            {([
              ["Impact", item.impact || "N/A", (IMPACT_COLOR as Record<string, string>)[item.impact || ""] || "#94a3b8"],
              ["Effort", item.effort || "N/A", (EFFORT_COLOR as Record<string, string>)[item.effort || ""] || "#94a3b8"],
              ["Timeline", (item.timeline as string) || "N/A", "#c9a84c"],
            ] as [string, string, string][]).map(([label, val, color]) => (
              <div key={label} style={{ background: "#fff", border: "1px solid #e8e0d4", padding: "12px 16px" }}>
                <div style={{ fontSize: "9px", color: "#9a8a6a", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "4px" }}>{label}</div>
                <div style={{ fontSize: "14px", fontWeight: 700, color, fontFamily: "Georgia, serif" }}>{val}</div>
              </div>
            ))}
          </div>
          <div style={{ background: "#fff8ee", border: "1px solid #c9a84c30", padding: "14px 16px", marginBottom: "12px" }}>
            <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "4px" }}>Expected Outcome</div>
            <div style={{ fontSize: "13px", color: "#3d2e1a", fontWeight: 600 }}>{item.roi_signal}</div>
          </div>
          <div style={{ background: "#1a1208", padding: "14px 16px" }}>
            <div style={{ fontSize: "10px", color: "#c9a84c80", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "4px" }}>First Step</div>
            <div style={{ fontSize: "13px", color: "#f5edd8", lineHeight: 1.6 }}>→ {item.first_step}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────

export default function AIIntegrationConsultant() {
  const [phase, setPhase] = useState("welcome"); // welcome | discovery | generating | report
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [report, setReport] = useState<ReportData | null>(null);
  const [activeTab, setActiveTab] = useState("integrations");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streamText]);

  const startDiscovery = async () => {
    setPhase("discovery");
    setLoading(true);
    const opener = [{ role: "user", content: "Hi, I'd like to explore how AI can be integrated into my business." }];
    setMessages([{ role: "user", content: "Hi, I'd like to explore how AI can be integrated into my business." }]);
    let aiText = "";
    await streamMessage(opener, DISCOVERY_SYSTEM, (chunk) => { setStreamText(chunk); aiText = chunk; });
    setMessages([
      { role: "user", content: "Hi, I'd like to explore how AI can be integrated into my business." },
      { role: "assistant", content: aiText },
    ]);
    setStreamText("");
    setLoading(false);
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userText = input.trim();
    setInput("");
    const newMessages = [...messages, { role: "user", content: userText }];
    setMessages(newMessages);
    setLoading(true);

    const apiMessages = newMessages.map(m => ({ role: m.role, content: m.content }));
    let aiText = "";
    await streamMessage(apiMessages, DISCOVERY_SYSTEM, (chunk) => { setStreamText(chunk); aiText = chunk; });

    const isReady = aiText.includes("[READY_FOR_REPORT]");
    const cleanText = aiText.replace("[READY_FOR_REPORT]", "").trim();

    const finalMessages = [...newMessages, { role: "assistant", content: cleanText }];
    setMessages(finalMessages);
    setStreamText("");
    setLoading(false);

    if (isReady) {
      setTimeout(() => triggerReport(finalMessages), 400);
    }
  };

  const triggerReport = async (msgs: Message[]) => {
    setPhase("generating");
    const convo = msgs.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n\n");
    const result = await generateReport(convo);
    setReport(result);
    setPhase("report");
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const integrations = (report?.integrations || []) as IntegrationItem[];
  const highImpactCount = integrations.filter((i) => i?.impact === "High").length;
  const quickTurnaroundCount = integrations.filter((i) => i?.effort === "Low").length;

  // ── WELCOME ─────────────────────────────────────────────
  if (phase === "welcome") return (
    <div style={{ minHeight: "100vh", background: "#f5f0e8", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "Georgia, serif", padding: "40px 20px" }}>
      <div style={{ maxWidth: "560px", width: "100%", textAlign: "center" }}>
        <div style={{ fontSize: "11px", color: "#9a8a6a", letterSpacing: "0.3em", textTransform: "uppercase", marginBottom: "32px" }}>AI Integration Consulting</div>
        <div style={{
          width: "80px", height: "80px", background: "#1a1208", margin: "0 auto 32px",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "36px", color: "#c9a84c",
          boxShadow: "0 20px 60px rgba(26,18,8,0.2)",
        }}>⟳</div>
        <h1 style={{ fontSize: "36px", fontWeight: 400, color: "#1a1208", margin: "0 0 16px", lineHeight: 1.2, letterSpacing: "-0.02em" }}>
          Find out exactly which<br />AI systems belong<br />in your business.
        </h1>
        <p style={{ fontSize: "15px", color: "#6b5b3e", lineHeight: 1.8, margin: "0 0 40px" }}>
          We&apos;ll ask you a few questions about what your company does,
          then generate a custom AI integration roadmap — specific
          tools, timelines, and first steps.
        </p>
        <button
          title="Start Discovery"
          onClick={startDiscovery}
          style={{
            background: "#1a1208", color: "#c9a84c",
            border: "none", padding: "16px 40px",
            fontSize: "13px", letterSpacing: "0.2em", textTransform: "uppercase",
            cursor: "pointer", fontFamily: "Georgia, serif",
            boxShadow: "0 8px 30px rgba(26,18,8,0.3)",
            transition: "all 0.2s",
          }}
          onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => { (e.target as HTMLButtonElement).style.background = "#2d2010"; (e.target as HTMLButtonElement).style.transform = "translateY(-2px)"; }}
          onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => { (e.target as HTMLButtonElement).style.background = "#1a1208"; (e.target as HTMLButtonElement).style.transform = "none"; }}
        >
          Start Free Assessment →
        </button>
        <p style={{ fontSize: "11px", color: "#b0a090", marginTop: "20px", letterSpacing: "0.05em" }}>Takes about 3 minutes · No signup required</p>
      </div>
    </div>
  );

  // ── DISCOVERY ────────────────────────────────────────────
  if (phase === "discovery") return (
    <div style={{ minHeight: "100vh", background: "#f5f0e8", fontFamily: "Georgia, serif", display: "flex", flexDirection: "column" }}>
      <div style={{ borderBottom: "1px solid #e0d8cc", padding: "16px 28px", display: "flex", alignItems: "center", gap: "14px", background: "#fff" }}>
        <div style={{ width: "36px", height: "36px", background: "#1a1208", display: "flex", alignItems: "center", justifyContent: "center", color: "#c9a84c", fontSize: "16px" }}>⟳</div>
        <div>
          <div style={{ fontSize: "14px", fontWeight: 700, color: "#1a1208", letterSpacing: "0.05em" }}>AI Integration Assessment</div>
          <div style={{ fontSize: "11px", color: "#9a8a6a", letterSpacing: "0.1em" }}>Discovery Conversation</div>
        </div>
        <div style={{ marginLeft: "auto", fontSize: "11px", color: "#9a8a6a", letterSpacing: "0.1em" }}>
          {messages.filter(m => m.role === "user").length > 1 && (
            <span style={{ color: "#c9a84c" }}>● </span>
          )}
          {messages.filter(m => m.role === "user").length}/4 responses
        </div>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "32px 28px", maxWidth: "680px", margin: "0 auto", width: "100%" }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            marginBottom: "24px",
            display: "flex", flexDirection: "column",
            alignItems: m.role === "user" ? "flex-end" : "flex-start",
          }}>
            {m.role === "assistant" && (
              <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "6px" }}>AI Consultant</div>
            )}
            <div style={{
              maxWidth: "85%",
              background: m.role === "user" ? "#1a1208" : "#fff",
              color: m.role === "user" ? "#f5edd8" : "#3d2e1a",
              padding: "16px 20px", fontSize: "14px", lineHeight: 1.75,
              border: m.role === "user" ? "none" : "1px solid #e0d8cc",
              boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
            }}>{m.content}</div>
          </div>
        ))}
        {streamText && (
          <div style={{ marginBottom: "24px", display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
            <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "6px" }}>AI Consultant</div>
            <div style={{ maxWidth: "85%", background: "#fff", color: "#3d2e1a", padding: "16px 20px", fontSize: "14px", lineHeight: 1.75, border: "1px solid #e0d8cc", boxShadow: "0 2px 12px rgba(0,0,0,0.06)" }}>
              {streamText}<span style={{ animation: "blink 1s infinite", color: "#c9a84c" }}>▋</span>
            </div>
          </div>
        )}
        {loading && !streamText && (
          <div style={{ display: "flex", gap: "6px", padding: "8px 0", marginLeft: "4px" }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{ width: "6px", height: "6px", background: "#c9a84c", borderRadius: "50%", animation: `bounce 1s ${i * 0.2}s infinite` }} />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div style={{ borderTop: "1px solid #e0d8cc", padding: "16px 28px", background: "#fff" }}>
        <div style={{ maxWidth: "680px", margin: "0 auto", display: "flex", gap: "12px" }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Type your response..."
            rows={2}
            disabled={loading}
            style={{
              flex: 1, background: "#faf8f5", border: "1px solid #e0d8cc",
              color: "#3d2e1a", fontSize: "14px", padding: "12px 16px",
              fontFamily: "Georgia, serif", resize: "none", outline: "none",
              lineHeight: 1.6,
            }}
          />
          <button
            title="Send Message"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              background: input.trim() && !loading ? "#1a1208" : "#e0d8cc",
              color: input.trim() && !loading ? "#c9a84c" : "#9a8a6a",
              border: "none", padding: "12px 20px",
              fontSize: "18px", cursor: input.trim() && !loading ? "pointer" : "default",
              transition: "all 0.15s",
            }}
          >→</button>
        </div>
      </div>
      <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0}} @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}`}</style>
    </div>
  );

  // ── GENERATING ───────────────────────────────────────────
  if (phase === "generating") return (
    <div style={{ minHeight: "100vh", background: "#f5f0e8", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "Georgia, serif" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "48px", color: "#c9a84c", marginBottom: "24px", animation: "spin 2s linear infinite", display: "inline-block" }}>⟳</div>
        <div style={{ fontSize: "18px", color: "#1a1208", fontWeight: 400, marginBottom: "8px" }}>Analyzing your business</div>
        <div style={{ fontSize: "13px", color: "#9a8a6a" }}>Mapping AI opportunities to your workflows...</div>
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  // ── REPORT ───────────────────────────────────────────────
  if (phase === "report" && report) {
    return (
      <div style={{ minHeight: "100vh", background: "#f5f0e8", fontFamily: "Georgia, serif" }}>
        {/* Header */}
        <div style={{ background: "#1a1208", padding: "32px 40px" }}>
          <div style={{ maxWidth: "900px", margin: "0 auto" }}>
            <div style={{ fontSize: "10px", color: "#c9a84c80", letterSpacing: "0.3em", textTransform: "uppercase", marginBottom: "12px" }}>AI Integration Report</div>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "20px" }}>
              <div>
                <h1 style={{ fontSize: "28px", fontWeight: 400, color: "#f5edd8", margin: "0 0 8px", lineHeight: 1.3 }}>{report.company_summary}</h1>
                <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
                  <span style={{ fontSize: "12px", color: "#c9a84c", background: "#c9a84c18", border: "1px solid #c9a84c40", padding: "4px 12px" }}>{report.industry}</span>
                  <span style={{ fontSize: "12px", color: "#9a8a6a" }}>{report.integrations?.length || 0} Integration Opportunities Identified</span>
                </div>
              </div>
              <button
                title="New Assessment"
                onClick={() => { setPhase("discovery"); setMessages([]); setReport(null); }}
                style={{ background: "transparent", border: "1px solid #c9a84c40", color: "#c9a84c80", padding: "8px 16px", fontSize: "11px", cursor: "pointer", letterSpacing: "0.1em", fontFamily: "Georgia, serif", flexShrink: 0 }}
              >New Assessment</button>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ background: "#fff", borderBottom: "1px solid #e0d8cc" }}>
          <div style={{ maxWidth: "900px", margin: "0 auto", display: "flex", gap: "0" }}>
            {[["integrations", "AI Integrations"], ["summary", "Strategic Summary"], ["quickwins", "Quick Wins"]].map(([id, label]) => (
              <button key={id} onClick={() => setActiveTab(id)} style={{
                background: "transparent", border: "none", borderBottom: `2px solid ${activeTab === id ? "#c9a84c" : "transparent"}`,
                color: activeTab === id ? "#1a1208" : "#9a8a6a", padding: "16px 24px",
                fontSize: "13px", cursor: "pointer", fontFamily: "Georgia, serif",
                fontWeight: activeTab === id ? 700 : 400, letterSpacing: "0.05em",
                transition: "all 0.15s",
              }}>{label}</button>
            ))}
          </div>
        </div>

        <div style={{ maxWidth: "900px", margin: "0 auto", padding: "32px 40px" }}>

          {/* INTEGRATIONS TAB */}
          {activeTab === "integrations" && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", marginBottom: "32px" }}>
                {([
                  ["Total Opportunities", integrations.length || 0, "#1a1208"],
                  ["High Impact", highImpactCount || 0, "#22c55e"],
                  ["Quick Turnaround", quickTurnaroundCount || 0, "#c9a84c"],
                ] as [string, number, string][]).map(([label, val, color]) => (
                  <div key={label} style={{ background: "#fff", border: "1px solid #e0d8cc", padding: "20px 24px" }}>
                    <div style={{ fontSize: "32px", fontWeight: 400, color, marginBottom: "4px" }}>{val}</div>
                    <div style={{ fontSize: "11px", color: "#9a8a6a", textTransform: "uppercase", letterSpacing: "0.15em" }}>{label}</div>
                  </div>
                ))}
              </div>
              {integrations.map((item, i) => <IntegrationCard key={String(item?.id || i)} item={item} index={i} />)}
            </div>
          )}

          {/* SUMMARY TAB */}
          {activeTab === "summary" && (
            <div style={{ display: "grid", gap: "20px" }}>
              <div style={{ background: "#fff", border: "1px solid #e0d8cc", padding: "28px 32px" }}>
                <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "16px" }}>Key Pain Points Identified</div>
                {(report.key_pain_points || []).map((p: string, i: number) => (
                  <div key={i} style={{ display: "flex", gap: "12px", marginBottom: "12px", alignItems: "flex-start" }}>
                    <span style={{ color: "#c9a84c", fontSize: "14px", flexShrink: 0, marginTop: "2px" }}>◈</span>
                    <span style={{ fontSize: "14px", color: "#3d2e1a", lineHeight: 1.6 }}>{p}</span>
                  </div>
                ))}
              </div>
              <div style={{ background: "#1a1208", border: "1px solid #2d2010", padding: "28px 32px" }}>
                <div style={{ fontSize: "10px", color: "#c9a84c80", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "12px" }}>Where to Start</div>
                <div style={{ fontSize: "15px", color: "#f5edd8", lineHeight: 1.8 }}>{report.recommended_start}</div>
              </div>
              <div style={{ background: "#fff8ee", border: "1px solid #c9a84c30", padding: "28px 32px" }}>
                <div style={{ fontSize: "10px", color: "#9a8a6a", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "12px" }}>Total Opportunity</div>
                <div style={{ fontSize: "15px", color: "#3d2e1a", lineHeight: 1.8 }}>{report.total_opportunity}</div>
              </div>
            </div>
          )}

          {/* QUICK WINS TAB */}
          {activeTab === "quickwins" && (
            <div>
              <div style={{ fontSize: "14px", color: "#6b5b3e", lineHeight: 1.8, marginBottom: "28px" }}>
                These are the fastest, lowest-effort AI integrations you can ship — often within days, not months.
              </div>
              {(report.quick_wins || []).map((win: string, i: number) => (
                <div key={i} style={{ background: "#fff", border: "1px solid #e0d8cc", padding: "20px 24px", marginBottom: "12px", display: "flex", gap: "16px", alignItems: "flex-start" }}>
                  <div style={{ width: "32px", height: "32px", background: "#c9a84c18", border: "1px solid #c9a84c40", display: "flex", alignItems: "center", justifyContent: "center", color: "#c9a84c", fontWeight: 700, fontSize: "13px", flexShrink: 0 }}>{i + 1}</div>
                  <div style={{ fontSize: "14px", color: "#3d2e1a", lineHeight: 1.7, paddingTop: "4px" }}>{win}</div>
                </div>
              ))}
              <div style={{ marginTop: "24px", background: "#1a1208", padding: "20px 24px" }}>
                <div style={{ fontSize: "10px", color: "#c9a84c80", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: "8px" }}>Ready to build?</div>
                <div style={{ fontSize: "14px", color: "#f5edd8", lineHeight: 1.7 }}>
                  Each of these integrations can be scoped, designed, and delivered. The recommended first step for each is in the full Integration Report.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}
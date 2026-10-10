import { Segmented } from "antd";
import type { Turn } from "../state/conversation";
import type { DocumentsState } from "../state/documents";
import { EvidenceList } from "./EvidenceList";
import { MaterialsPanel } from "./MaterialsPanel";
import { StepTrace } from "./StepTrace";
import styles from "./RightRail.module.css";

export const USAGE =
  "右栏抽屉：依据 / 轨迹 / 材料三个页签共用一个面板；依据与轨迹跟随当前聚焦的一轮，材料是全局台账。";

export type RailTab = "evidence" | "trace" | "materials";

type Props = {
  tab: RailTab;
  onTab: (tab: RailTab) => void;
  turn: Turn | null;
  highlight: number | null;
  documents: DocumentsState;
  activeDocIds: string[];
  onToggleDoc: (docId: string) => void;
};

const TITLES: Record<RailTab, string> = {
  evidence: "依据",
  trace: "轨迹",
  materials: "材料",
};

export function RightRail({ tab, onTab, turn, highlight, documents, activeDocIds, onToggleDoc }: Props) {
  return (
    <aside className={styles.rail}>
      <div className={styles.panel}>
        <div className={styles.head}>
          <div className={styles.titleRow}>
            <span className={styles.title}>{TITLES[tab]}</span>
            {tab !== "materials" && turn ? (
              <span className={styles.subtitle} title={turn.question}>
                {turn.question}
              </span>
            ) : null}
          </div>
          <Segmented
            size="small"
            value={tab}
            onChange={(value) => onTab(value as RailTab)}
            options={[
              { label: "依据", value: "evidence" },
              { label: "轨迹", value: "trace" },
              { label: "材料", value: "materials" },
            ]}
          />
        </div>

        <div className={`${styles.body} scrollbar`}>
          {tab === "evidence" ? <EvidenceList answer={turn?.answer ?? null} highlight={highlight} /> : null}
          {tab === "trace" ? (
            <StepTrace
              variant="panel"
              steps={turn?.steps ?? []}
              streaming={turn?.status === "streaming"}
              mode={turn?.mode ?? "stream"}
              elapsedMs={turn?.elapsedMs ?? null}
              question={turn?.question ?? null}
            />
          ) : null}
          {tab === "materials" ? (
            <MaterialsPanel documents={documents} activeDocIds={activeDocIds} onToggle={onToggleDoc} />
          ) : null}
        </div>
      </div>
    </aside>
  );
}

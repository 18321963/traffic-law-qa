import { useEffect, useState } from "react";
import { Button, Drawer } from "antd";
import { ReadOutlined } from "@ant-design/icons";
import type { DonePayload, Evidence, RetrievedArticle } from "../api/types";
import styles from "./EvidenceList.module.css";

export const USAGE =
  "右栏依据页签：答案引用的依据做成卡片（法名 · 条号 + 摘录 + 阅读原文抽屉），下面折叠本次检索的全部命中与通道口径。";

type Props = {
  answer: DonePayload | null;
  highlight: number | null;
};

type Reading = {
  title: string;
  label: string;
  text: string;
  meta: string;
};

function HitRow({ row, index }: { row: RetrievedArticle; index: number }) {
  const article = row.article;
  const lawName = article?.law_name ?? row.citation ?? "（未给法名）";
  const articleNo = article?.article_no ?? "";
  const text = article?.text ?? "";
  const ranks: string[] = [];
  if (typeof row.vector_rank === "number") ranks.push(`向量 #${row.vector_rank}`);
  if (typeof row.bm25_rank === "number") ranks.push(`BM25 #${row.bm25_rank}`);
  return (
    <details className={styles.hit}>
      <summary>
        <span className={styles.hitIndex}>{index + 1}</span>
        <span className={styles.hitTitle}>
          {lawName}
          {articleNo ? ` · ${articleNo}` : ""}
        </span>
        <span className={styles.hitRanks}>{ranks.join(" ")}</span>
      </summary>
      {text ? <p className={styles.statute}>{text}</p> : <p className={styles.hitEmpty}>这条命中没有带回正文</p>}
    </details>
  );
}

export function EvidenceList({ answer, highlight }: Props) {
  const [reading, setReading] = useState<Reading | null>(null);
  const evidences: Evidence[] = answer?.evidences ?? [];
  const hits = answer?.retrieval?.articles ?? [];

  useEffect(() => {
    if (highlight === null) return;
    const node = document.getElementById(`ev-card-${highlight}`);
    node?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [highlight, answer]);

  if (evidences.length === 0 && hits.length === 0) {
    return <p className={styles.empty}>这一轮还没有可看的依据；提问之后，答案引用的条文会出现在这里。</p>;
  }

  return (
    <div className={styles.evidence}>
      {evidences.length > 0 ? (
        <>
          <div className={styles.sectionTitle}>答案引用的依据</div>
          <div className={styles.cards}>
            {evidences.map((row, index) => (
              <div
                key={`${row.label}-${index}`}
                id={`ev-card-${index + 1}`}
                className={`${styles.card} ${highlight === index + 1 ? styles.isHighlight : ""}`}
              >
                <div className={styles.cardHead}>
                  <span className={styles.cardName} title={row.citation}>
                    {row.citation}
                  </span>
                  <span className={styles.cardIndex}>{row.label}</span>
                </div>
                <p className={styles.cardExcerpt}>{row.text}</p>
                <div className={styles.cardFoot}>
                  <span className={styles.cardScore}>
                    {typeof row.score === "number" ? `相关度 ${row.score.toFixed(3)}` : "未给相关度"}
                  </span>
                  <Button
                    size="small"
                    className={styles.readBtn}
                    icon={<ReadOutlined />}
                    onClick={() =>
                      setReading({
                        title: row.citation,
                        label: row.label,
                        text: row.text,
                        meta:
                          typeof row.score === "number"
                            ? `相关度 ${row.score.toFixed(3)}`
                            : "本次未给出相关度分数",
                      })
                    }
                  >
                    阅读原文
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className={styles.empty}>答案没有引用依据。</p>
      )}

      {hits.length > 0 ? (
        <details className={styles.hits}>
          <summary>本次检索的 {hits.length} 条命中（含未被引用的）</summary>
          <div className={styles.hitsBody}>
            {hits.map((row, index) => (
              <HitRow key={row.parent_id ?? index} row={row} index={index} />
            ))}
          </div>
        </details>
      ) : null}

      {answer && typeof answer.retrieval?.used_vector === "boolean" ? (
        <p className={styles.channels}>
          本次通道：{answer.retrieval.used_vector ? "稠密" : ""}
          {answer.retrieval.used_vector && answer.retrieval.used_bm25 ? " + " : ""}
          {answer.retrieval.used_bm25 ? "BM25" : ""}
          {!answer.retrieval.used_vector && !answer.retrieval.used_bm25 ? "无（未检索）" : ""}
        </p>
      ) : null}

      <Drawer
        title={reading?.title ?? ""}
        width={520}
        open={reading !== null}
        onClose={() => setReading(null)}
        styles={{ body: { paddingTop: 16 } }}
      >
        {reading ? (
          <div className={styles.reading}>
            <p className={styles.readingMeta}>
              {reading.label} · {reading.meta}
            </p>
            <p className={styles.readingText}>{reading.text}</p>
          </div>
        ) : null}
      </Drawer>
    </div>
  );
}

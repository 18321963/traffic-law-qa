import type { DonePayload, Evidence, RetrievedArticle } from "../api/types";

type Props = {
  answer: DonePayload | null;
  anchorPrefix: string;
  highlight: number | null;
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
    <details className="hit">
      <summary>
        <span className="hit__index">{index + 1}</span>
        <span className="hit__title">
          {lawName}
          {articleNo ? ` · ${articleNo}` : ""}
        </span>
        <span className="hit__ranks">{ranks.join(" ")}</span>
      </summary>
      {text ? <p className="statute">{text}</p> : <p className="hit__empty">这条命中没有带回正文</p>}
    </details>
  );
}

export function EvidenceList({ answer, anchorPrefix, highlight }: Props) {
  const evidences: Evidence[] = answer?.evidences ?? [];
  const hits = answer?.retrieval?.articles ?? [];
  if (evidences.length === 0 && hits.length === 0) return null;
  return (
    <section className="evidence">
      {evidences.length > 0 ? (
        <>
          <h4 className="evidence__title">答案引用的依据</h4>
          <ol className="evidence__list">
            {evidences.map((row, index) => (
              <li
                key={`${row.label}-${index}`}
                id={`${anchorPrefix}-ev-${index + 1}`}
                className={`evidence__item${highlight === index + 1 ? " is-highlight" : ""}`}
              >
                <div className="evidence__label">{row.label}</div>
                <div className="evidence__main">
                  <p className="evidence__citation">{row.citation}</p>
                  <p className="statute">{row.text}</p>
                  {typeof row.score === "number" ? (
                    <p className="evidence__score">相关度 {row.score.toFixed(3)}</p>
                  ) : null}
                </div>
              </li>
            ))}
          </ol>
        </>
      ) : (
        <h4 className="evidence__title">答案没有引用依据</h4>
      )}
      {hits.length > 0 ? (
        <details className="hits">
          <summary>本次检索的 {hits.length} 条命中（含未被引用的）</summary>
          <div className="hits__body">
            {hits.map((row, index) => (
              <HitRow key={row.parent_id ?? index} row={row} index={index} />
            ))}
          </div>
        </details>
      ) : null}
      {answer && typeof answer.retrieval?.used_vector === "boolean" ? (
        <p className="evidence__channels">
          本次通道：{answer.retrieval.used_vector ? "稠密" : ""}
          {answer.retrieval.used_vector && answer.retrieval.used_bm25 ? " + " : ""}
          {answer.retrieval.used_bm25 ? "BM25" : ""}
          {!answer.retrieval.used_vector && !answer.retrieval.used_bm25 ? "无（未检索）" : ""}
        </p>
      ) : null}
    </section>
  );
}

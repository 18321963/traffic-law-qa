import type { Evidence, ParentChunk, RetrievedArticle, StepEvent } from "../types";

export const USAGE =
  "示例数据取材自仓内 法规知识库/text 的真实条文（条号与原文一致），仅作界面演练，不代表服务的真实回答；每条答案的 notes 里都写明这一点。";

export const NOTE_MOCK = "示例数据：本回答由界面内置夹具生成，不是服务检索的结果";

export const CLARIFY_MESSAGE =
  "这个问题涉及具体的地区，先确认按哪里的规定回答：回复一个地区名（如「深圳」）就按该地区法规加上全国法作答；回复 national 则只按全国法作答。";

export const UNAVAILABLE_ANSWER = "（未配置大模型，下面只给出召回的法条）";

export const UNAVAILABLE_NOTE = "未配置 LLM_API_KEY，无法生成答案（可用 search 查看检索结果）";

export const EMPTY_ANSWER =
  "本次检索没有命中可用法条，因此不作答。可以换个说法再问，或直接点名法规（如《中华人民共和国道路交通安全法》《深圳经济特区道路交通安全违法行为处罚条例》）。";

export const EMPTY_NOTE = "检索结果为空：未命中任何法条";

export const LOCAL_LAWS = ["深圳经济特区道路交通安全违法行为处罚条例", "深圳经济特区智能网联汽车管理条例"];

export const LAW_COUNT = 8;

type ArticleSpec = {
  id: string;
  law: string;
  article_no: string;
  text: string;
  local?: boolean;
};

export const ARTICLES: Record<string, ArticleSpec> = {
  demerit10: {
    id: "demerit-points-10",
    law: "道路交通安全违法行为记分管理办法",
    article_no: "第十条",
    text:
      "机动车驾驶人有下列交通违法行为之一，一次记6分：……（八）驾驶机动车不按交通信号灯指示通行的；……",
  },
  law26: {
    id: "road-traffic-safety-law-26",
    law: "中华人民共和国道路交通安全法",
    article_no: "第二十六条",
    text: "交通信号灯由红灯、绿灯、黄灯组成。红灯表示禁止通行，绿灯表示准许通行，黄灯表示警示。",
  },
  law90: {
    id: "road-traffic-safety-law-90",
    law: "中华人民共和国道路交通安全法",
    article_no: "第九十条",
    text: "机动车驾驶人违反道路交通安全法律、法规关于道路通行规定的，处警告或者二十元以上二百元以下罚款。本法另有规定的，依照规定处罚。",
  },
  law91: {
    id: "road-traffic-safety-law-91",
    law: "中华人民共和国道路交通安全法",
    article_no: "第九十一条",
    text:
      "饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。因饮酒后驾驶机动车被处罚，再次饮酒后驾驶机动车的，处十日以下拘留，并处一千元以上二千元以下罚款，吊销机动车驾驶证。醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。饮酒后驾驶营运机动车的，处十五日拘留，并处五千元罚款，吊销机动车驾驶证，五年内不得重新取得机动车驾驶证。醉酒驾驶营运机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；十年内不得重新取得机动车驾驶证，重新取得机动车驾驶证后，不得驾驶营运机动车。",
  },
  demerit8: {
    id: "demerit-points-8",
    law: "道路交通安全违法行为记分管理办法",
    article_no: "第八条",
    text: "机动车驾驶人有下列交通违法行为之一，一次记12分：（一）饮酒后驾驶机动车的；……",
  },
  reg71: {
    id: "road-traffic-safety-regulation-71",
    law: "中华人民共和国道路交通安全法实施条例",
    article_no: "第七十一条",
    text:
      "非机动车载物，应当遵守下列规定：（一）自行车、电动自行车、残疾人机动轮椅车载物，高度从地面起不得超过1.5米，宽度左右各不得超出车把0.15米，长度前端不得超出车轮，后端不得超出车身0.3米；……",
  },
  reg55: {
    id: "road-traffic-safety-regulation-55",
    law: "中华人民共和国道路交通安全法实施条例",
    article_no: "第五十五条",
    text: "机动车载人应当遵守下列规定：……（三）摩托车后座不得乘坐未满12周岁的未成年人，轻便摩托车不得载人。",
  },
  sz9: {
    id: "sz-traffic-penalty-9",
    law: "深圳经济特区道路交通安全违法行为处罚条例",
    article_no: "第九条",
    local: true,
    text:
      "驾驶非机动车有下列行为之一的，处五百元罚款：（一）驾驶改装、加装动力装置和不符合国家技术标准的灯光装置的非机动车上道路行驶的；（二）违反规定在机动车道内行驶的；……",
  },
  sz10: {
    id: "sz-traffic-penalty-10",
    law: "深圳经济特区道路交通安全违法行为处罚条例",
    article_no: "第十条",
    local: true,
    text: "驾驶电动自行车不按交通信号、标识规定通行或者逆行的处三百元罚款。饮酒后驾驶电动自行车的，处五百元以上二千元以下罚款。",
  },
};

export function citationOf(spec: ArticleSpec): string {
  return `《${spec.law}》${spec.article_no}`;
}

export function evidence(label: string, spec: ArticleSpec, score: number): Evidence {
  return { label, citation: citationOf(spec), text: spec.text, score };
}

export function chunkOf(spec: ArticleSpec): ParentChunk {
  return {
    parent_id: spec.id,
    law_id: spec.id.replace(/-\d+$/, ""),
    law_name: spec.law,
    version: "示例版本",
    citation: citationOf(spec),
    article_no: spec.article_no,
    text: spec.text,
  };
}

export function hitOf(spec: ArticleSpec, score: number, vectorRank: number, bm25Rank: number): RetrievedArticle {
  return {
    parent_id: spec.id,
    citation: citationOf(spec),
    article: chunkOf(spec),
    score,
    vector_rank: vectorRank,
    bm25_rank: bm25Rank,
    vector_score: Math.round(score * 1000) / 1000,
    bm25_score: Math.round((score * 300 + 4) * 10) / 10,
    hit_chunks: [spec.text.slice(0, 40)],
    law_hint: null,
  };
}

export type CaseId = "red_light" | "shenzhen" | "shenzhen_national" | "drunk" | "empty" | "degraded" | "mismatch";

export type CaseContent = {
  question: string;
  draft: string;
  final: string;
  evidences: Evidence[];
  articles: RetrievedArticle[];
  review: null | {
    passed: boolean;
    score: number;
    threshold: number;
    total: number;
    supported: number;
    unsupported: string[];
    model: string;
    original_text: string;
  };
  model: string;
  notes: string[];
  fallback: boolean;
};

const RED_LIGHT_ANSWER = [
  "驾驶机动车不按交通信号灯指示通行的，一次记 6 分[依据1]。",
  "",
  "《道路交通安全违法行为记分管理办法》把「驾驶机动车不按交通信号灯指示通行的」列在第十条「一次记6分」的情形里[依据1]；红灯本身的含义是全国统一的——红灯表示禁止通行[依据2]。",
  "",
  "记分之外还有罚款：《中华人民共和国道路交通安全法》第九十条规定，机动车驾驶人违反道路通行规定的，处警告或者二十元以上二百元以下罚款[依据3]。",
  "",
  "两点提示：记分记在驾驶人身上，一次违法与一次处罚对应；路口电子设备的取证流程与申诉渠道，以当地公安机关交通管理部门的说明为准。",
].join("\n");

const SHENZHEN_ANSWER = [
  "深圳经济特区把电动自行车按非机动车管理，处罚依据在《深圳经济特区道路交通安全违法行为处罚条例》[依据1][依据2]：",
  "",
  "驾驶电动自行车不按交通信号、标识规定通行或者逆行的，处三百元罚款；饮酒后驾驶电动自行车的，处五百元以上二千元以下罚款[依据1]。驾驶改装、加装动力装置和不符合国家技术标准的灯光装置的非机动车上道路行驶，或者违反规定在机动车道内行驶的，处五百元罚款[依据2]。",
  "",
  "关于载人：本次检索到的依据里没有直接规定「电动自行车载人」的条款。相邻的全国性规定是《中华人民共和国道路交通安全法实施条例》第七十一条（非机动车载物）与第五十五条（机动车载人）[依据3]，两者都不是针对电动自行车载人的专门条款。",
  "",
  "因此，载人这一具体口径建议向深圳市公安机关交通管理部门确认；上面两条本地条款是本次检索能给出的确定依据。",
].join("\n");

const NATIONAL_ONLY_ANSWER = [
  "只按全国法作答：本次检索到的依据里没有专门规定「电动自行车载人」的条款[依据1][依据2]。",
  "",
  "相邻的两条是《中华人民共和国道路交通安全法实施条例》第七十一条（非机动车载物，含电动自行车载物高度、宽度、长度限制）[依据1]与第五十五条（机动车载人，含摩托车后座限制）[依据2]——载物与载人是两回事，两条都不能直接当作电动自行车载人的处罚依据。",
  "",
  "如果要地方口径，可以在地区选择里选「深圳」再问一次，届时会带上深圳经济特区的法规。",
].join("\n");

const DRUNK_ANSWER = [
  "醉酒驾驶机动车的，处理分三层：约束、吊销、追刑[依据1]。",
  "",
  "第一层是当场处置：由公安机关交通管理部门约束至酒醒。第二层是驾驶证：吊销机动车驾驶证，五年内不得重新取得。第三层是刑事：依法追究刑事责任，对应刑法上的危险驾驶罪[依据1]。",
  "",
  "如果开的是营运机动车，后果更重：醉酒驾驶营运机动车的，约束至酒醒、吊销驾驶证、依法追究刑事责任，十年内不得重新取得机动车驾驶证；重新取得后，也不得再驾驶营运机动车[依据1]。",
  "",
  "还有一条容易漏掉的记分口径：饮酒后驾驶机动车的，一次记 12 分[依据2]。注意「饮酒」与「醉酒」是两个不同的血液酒精含量区间，处罚也不同——饮酒后驾驶是暂扣六个月驾驶证加一千元以上二千元以下罚款[依据1]。",
  "",
  "补充两点常见疑问：其一，醉酒驾驶的认定数值（每 100 毫升血液酒精含量 80 毫克以上）与检验程序由专门规定给出，本知识库的这批依据里没有收录该数值条款，实际认定以办案机关的检验结论为准。其二，醉驾被吊销后重新申请驾驶证，还要受申请条件的限制，具体以《机动车驾驶证申领和使用规定》为准。",
  "",
  "以上是本次检索到依据范围内的答复；个案处理请以办案机关的法律文书为准。",
].join("\n");

function articleHits(specs: ArticleSpec[]): RetrievedArticle[] {
  return specs.map((spec, index) => hitOf(spec, 0.032 - index * 0.004, index + 1, index + 2));
}

const REVIEW_MODEL = "qwen-flash";

function reviewPassed(total: number, original: string): CaseContent["review"] {
  return {
    passed: true,
    score: 1,
    threshold: 0.35,
    total,
    supported: total,
    unsupported: [],
    model: REVIEW_MODEL,
    original_text: original,
  };
}

export function contentOf(caseId: CaseId): CaseContent {
  if (caseId === "red_light") {
    const specs = [ARTICLES.demerit10, ARTICLES.law26, ARTICLES.law90, ARTICLES.law91, ARTICLES.demerit8, ARTICLES.reg71];
    return {
      question: "",
      draft: RED_LIGHT_ANSWER,
      final: RED_LIGHT_ANSWER,
      evidences: [
        evidence("依据1", ARTICLES.demerit10, 0.032),
        evidence("依据2", ARTICLES.law26, 0.028),
        evidence("依据3", ARTICLES.law90, 0.021),
      ],
      articles: articleHits(specs),
      review: reviewPassed(3, RED_LIGHT_ANSWER),
      model: "qwen-flash",
      notes: [NOTE_MOCK],
      fallback: false,
    };
  }
  if (caseId === "shenzhen") {
    return {
      question: "",
      draft: SHENZHEN_ANSWER,
      final: SHENZHEN_ANSWER,
      evidences: [
        evidence("依据1", ARTICLES.sz10, 0.041),
        evidence("依据2", ARTICLES.sz9, 0.037),
        evidence("依据3", ARTICLES.reg71, 0.019),
      ],
      articles: articleHits([ARTICLES.sz10, ARTICLES.sz9, ARTICLES.reg71, ARTICLES.reg55, ARTICLES.law90, ARTICLES.law26]),
      review: reviewPassed(3, SHENZHEN_ANSWER),
      model: "qwen-flash",
      notes: [NOTE_MOCK],
      fallback: false,
    };
  }
  if (caseId === "shenzhen_national") {
    return {
      question: "",
      draft: NATIONAL_ONLY_ANSWER,
      final: NATIONAL_ONLY_ANSWER,
      evidences: [evidence("依据1", ARTICLES.reg71, 0.026), evidence("依据2", ARTICLES.reg55, 0.022)],
      articles: articleHits([ARTICLES.reg71, ARTICLES.reg55, ARTICLES.law90, ARTICLES.law26]),
      review: reviewPassed(2, NATIONAL_ONLY_ANSWER),
      model: "qwen-flash",
      notes: [NOTE_MOCK, "按「只按全国法」作答：本次未纳入地方性法规"],
      fallback: false,
    };
  }
  if (caseId === "drunk") {
    return {
      question: "",
      draft: DRUNK_ANSWER,
      final: DRUNK_ANSWER,
      evidences: [
        evidence("依据1", ARTICLES.law91, 0.044),
        evidence("依据2", ARTICLES.demerit8, 0.03),
      ],
      articles: articleHits([ARTICLES.law91, ARTICLES.demerit8, ARTICLES.law90, ARTICLES.reg55, ARTICLES.law26, ARTICLES.demerit10]),
      review: reviewPassed(2, DRUNK_ANSWER),
      model: "qwen-flash",
      notes: [NOTE_MOCK],
      fallback: false,
    };
  }
  if (caseId === "degraded") {
    return {
      question: "",
      draft: UNAVAILABLE_ANSWER,
      final: UNAVAILABLE_ANSWER,
      evidences: [],
      articles: articleHits([ARTICLES.law90, ARTICLES.law26, ARTICLES.demerit10]),
      review: null,
      model: "(unavailable)",
      notes: [UNAVAILABLE_NOTE, NOTE_MOCK],
      fallback: false,
    };
  }
  if (caseId === "mismatch") {
    const draft = [
      "驾驶机动车不按交通信号灯指示通行的，一次记 3 分[依据1]。",
      "",
      "路口未设非机动车信号灯时，非机动车按机动车信号灯通行。",
    ].join("\n");
    const final = [
      "本次回答未通过依据复核，暂不给出结论 —— 现有依据不足以支撑它，需人工复审。",
      "",
      "复核结果：支撑 1/3（阈值 0.35），需人工复审。",
      "",
      "候选法条（未经复核确认，仅供人工核对）：",
      "  【依据1】 《道路交通安全违法行为记分管理办法》(2021-12-27)第十条",
      "  【依据2】 《中华人民共和国道路交通安全法》第九十条",
      "  【依据3】 《中华人民共和国道路交通安全法实施条例》第七十一条",
    ].join("\n");
    return {
      question: "",
      draft,
      final,
      evidences: [
        evidence("依据1", ARTICLES.demerit10, 0.032),
        evidence("依据2", ARTICLES.law90, 0.024),
        evidence("依据3", ARTICLES.reg71, 0.019),
      ],
      articles: articleHits([ARTICLES.demerit10, ARTICLES.law90, ARTICLES.reg71, ARTICLES.law26]),
      review: {
        passed: false,
        score: 0.333,
        threshold: 0.35,
        total: 3,
        supported: 1,
        unsupported: ["依据2", "依据3"],
        model: REVIEW_MODEL,
        original_text: draft,
      },
      model: "qwen-flash",
      notes: [NOTE_MOCK, "复核未通过：支撑 1/3 < 阈值 0.35 —— 已降级为「依据不足」，需人工复审"],
      fallback: false,
    };
  }
  return {
    question: "",
    draft: EMPTY_ANSWER,
    final: EMPTY_ANSWER,
    evidences: [],
    articles: [],
    review: null,
    model: "(skip)",
    notes: [EMPTY_NOTE, NOTE_MOCK],
    fallback: false,
  };
}

export function stepRegion(region: string | null, laws: number): StepEvent {
  return { node: "region", region, laws };
}

export function stepClarify(region: string | null, laws: number): StepEvent {
  return { node: "clarify", region, laws };
}

export function stepAgent(turn: number, maxSteps: number, tools: string[], truncated = false): StepEvent {
  return { node: "agent", turn, max_steps: maxSteps, tools, truncated };
}

export function stepTools(retrievals: number, hits: number, materials = 0, web = 0): StepEvent {
  return { node: "tools", retrievals, hits, materials, web };
}

export function stepFinalize(evidenceCount: number, fallback = false): StepEvent {
  return { node: "finalize", evidence: evidenceCount, fallback };
}

export function stepReview(review: CaseContent["review"]): StepEvent {
  return review === null ? { node: "review" } : { node: "review", passed: review.passed, score: review.score };
}

export function splitDeltas(text: string): string[] {
  const sizes = [2, 1, 3, 2, 1, 4, 2, 3, 1, 2];
  const pieces: string[] = [];
  let cursor = 0;
  let index = 0;
  while (cursor < text.length) {
    const size = sizes[index % sizes.length];
    pieces.push(text.slice(cursor, cursor + size));
    cursor += size;
    index += 1;
  }
  return pieces;
}

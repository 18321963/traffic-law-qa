import { useId } from "react";

export const USAGE =
  "自绘品牌标记：圆角方块上放一盏三色信号灯，同时用作侧栏标志与回答者头像；不引外部图片素材。";

type Props = {
  size?: number;
};

export function BrandMark({ size = 28 }: Props) {
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const gradient = `bm-${id}`;
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" role="img" aria-label="交通法规问答">
      <defs>
        <linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0862fe" />
          <stop offset="1" stopColor="#639bff" />
        </linearGradient>
      </defs>
      <rect width="28" height="28" rx="9" fill={`url(#${gradient})`} />
      <rect x="9" y="4.5" width="10" height="19" rx="5" fill="#ffffff" opacity="0.94" />
      <circle cx="14" cy="9.5" r="2" fill="#ff5f57" />
      <circle cx="14" cy="14" r="2" fill="#febc2e" />
      <circle cx="14" cy="18.5" r="2" fill="#28c840" />
    </svg>
  );
}

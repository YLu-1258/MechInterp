import React from 'react';

interface TooltipProps {
  content: React.ReactNode;
  x: number;
  y: number;
  visible: boolean;
}

const Tooltip: React.FC<TooltipProps> = ({ content, x, y, visible }) => {
  if (!visible) return null;

  // Offset from cursor to avoid covering the target
  const offsetX = 12;
  const offsetY = -8;

  return (
    <div
      className="tooltip"
      style={{
        left: x + offsetX,
        top: y + offsetY,
        transform: 'translateY(-100%)',
        opacity: visible ? 1 : 0,
        transition: 'opacity 150ms ease',
      }}
    >
      {content}
    </div>
  );
};

export default React.memo(Tooltip);

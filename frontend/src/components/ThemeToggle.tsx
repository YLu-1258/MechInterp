import React from 'react';

interface ThemeToggleProps {
  isDark: boolean;
  onToggle: () => void;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ isDark, onToggle }) => {
  return (
    <div className="flex items-center gap-3">
      <span className="text-label">
        {isDark ? 'Dark' : 'Light'}
      </span>
      <button
        className="theme-switch"
        onClick={onToggle}
        aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        role="switch"
        aria-checked={!isDark}
      >
        <div className={`theme-switch-knob ${isDark ? '' : 'light'}`}>
          {isDark ? '🌙' : '☀️'}
        </div>
      </button>
    </div>
  );
};

export default React.memo(ThemeToggle);

import type { CSSProperties } from 'react';

export const gameReplayRootStyle: CSSProperties = {
  height: 'calc(100vh - var(--mobile-shell-offset, 0px))',
  background: 'var(--page-bg)',
  overflow: 'hidden',
};

export const centeredStatusStyle: CSSProperties = {
  background: 'var(--page-bg)',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 12,
};

export const mutedStatusTextStyle: CSSProperties = {
  color: 'var(--text-muted)',
  fontSize: 14,
};

export const errorStatusTextStyle: CSSProperties = {
  color: 'var(--error)',
  fontSize: 14,
};

export const backLinkButtonStyle: CSSProperties = {
  color: 'var(--accent)',
  fontSize: 14,
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  textDecoration: 'underline',
};

export const iconBtnStyle: CSSProperties = {
  padding: '3px 8px',
  borderRadius: 4,
  border: '1px solid var(--overlay-border)',
  background: 'var(--overlay-bg)',
  color: 'var(--control-muted)',
  fontSize: 12,
  cursor: 'pointer',
  flexShrink: 0,
  backdropFilter: 'blur(4px)',
};

export const smallBtnStyle: CSSProperties = {
  padding: 3,
  borderRadius: 4,
  border: '1px solid var(--overlay-border)',
  background: 'var(--overlay-bg)',
  color: 'var(--control-muted)',
  cursor: 'pointer',
  flexShrink: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backdropFilter: 'blur(4px)',
};

export const floatingSwitchBtnStyle: CSSProperties = {
  width: '100%',
  borderRadius: 6,
  border: '1px solid var(--overlay-border)',
  background: 'var(--overlay-bg)',
  padding: '6px 9px',
  textAlign: 'left',
  fontSize: 12,
  color: 'var(--control-muted)',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  backdropFilter: 'blur(4px)',
};

export const perspectiveToggleStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  borderRadius: 8,
  border: '1px solid var(--overlay-border)',
  background: 'var(--overlay-toolbar-bg)',
  color: 'var(--control-muted)',
  padding: '6px 10px',
  fontSize: 12,
  cursor: 'pointer',
  backdropFilter: 'blur(8px)',
};

export const perspectiveDrawerStyle: CSSProperties = {
  position: 'absolute',
  left: 0,
  bottom: 'calc(100% + 8px)',
  zIndex: 30,
  background: 'var(--overlay-bg)',
  border: '1px solid var(--overlay-border)',
  borderRadius: 8,
  padding: '10px 10px 8px',
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  minWidth: 170,
  boxShadow: '0 8px 24px rgba(0,0,0,0.28)',
  backdropFilter: 'blur(10px)',
};

export const floatingTopBarWrapStyle: CSSProperties = {
  position: 'absolute',
  top: 12,
  left: 168,
  right: 12,
  zIndex: 34,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: 8,
  pointerEvents: 'none',
};

export const toolbarToggleStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  padding: '5px 10px',
  borderRadius: 6,
  border: '1px solid var(--overlay-border)',
  background: 'var(--overlay-bg)',
  color: 'var(--control-muted)',
  fontSize: 12,
  cursor: 'pointer',
  backdropFilter: 'blur(6px)',
  pointerEvents: 'auto',
};

export const floatingTopBarStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  flexWrap: 'wrap',
  padding: '7px 10px',
  borderRadius: 8,
  width: 'min(980px, calc(100% - 32px))',
  background: 'var(--overlay-toolbar-bg)',
  border: '1px solid var(--overlay-toolbar-border)',
  backdropFilter: 'blur(8px)',
  pointerEvents: 'auto',
};

export const floatingPerspectiveStyle: CSSProperties = {
  position: 'absolute',
  left: 12,
  bottom: 12,
  zIndex: 30,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: 6,
};

export const sidePanelContainerStyle: CSSProperties = {
  width: 198,
  minWidth: 198,
  height: '100%',
  borderLeft: '1px solid var(--sidepanel-border)',
  background: 'var(--sidepanel-bg)',
  backdropFilter: 'blur(6px)',
};

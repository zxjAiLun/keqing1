/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 麻将主题色（教育与图表配色 #10）
        board: {
          felt: '#2a9d8f',
          feltDark: '#264653',
          wood: '#8b4513',
          woodLight: '#a0522d',
        },
        tile: {
          white: '#f8f8f0',
          ivory: '#faf0e6',
          red: '#c0392b',
          green: '#27ae60',
          blue: '#2980b9',
          gold: '#d4af37',
        },
        player: {
          east: '#e74c3c',
          south: '#3498db',
          west: '#2ecc71',
          north: '#9b59b6',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Microsoft YaHei', 'sans-serif'],
        mono: ['Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
}

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/Layout/MainLayout';
import { DashboardPage } from './pages/DashboardPage';
import { HomePage } from './pages/HomePage';
import { ReviewPage } from './pages/ReviewPage';
import { ReplayViewPage } from './pages/ReplayViewPage';
import { GameBoardPage } from './pages/GameBoardPage';
import { GameBoardReplayPage } from './pages/GameBoardReplayPage';
import { BattlePage } from './pages/BattlePage';
import { BotBattlePage } from './pages/BotBattlePage';
import { ThemeProvider } from './context/ThemeContext';
import './styles/globals.css';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<MainLayout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/battle" element={<BattlePage />} />
            <Route path="/bot-battle" element={<BotBattlePage />} />
            <Route path="/review" element={<ReviewPage />} />
            <Route path="/game" element={<GameBoardPage />} />
            <Route path="/game-replay" element={<GameBoardReplayPage />} />
            <Route path="/replay" element={<ReplayViewPage />} />
          </Route>
          <Route path="/home" element={<HomePage />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

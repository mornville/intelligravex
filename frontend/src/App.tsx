import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'
import BotsPage from './pages/BotsPage'
import BotDetailPage from './pages/BotDetailPage'
import KeysPage from './pages/KeysPage'
import ConversationsPage from './pages/ConversationsPage'
import ConversationDetailPage from './pages/ConversationDetailPage'
import DeveloperPage from './pages/DeveloperPage'
import LandingPage from './pages/LandingPage'

export default function App() {
  const location = useLocation()
  return (
    <AuthGate>
      <Layout>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<Navigate to="/bots" replace />} />
          <Route path="/bots" element={<BotsPage />} />
          <Route path="/bots/:botId" element={<BotDetailPage />} />
          <Route path="/keys" element={<KeysPage />} />
          <Route path="/developer" element={<DeveloperPage />} />
          <Route path="/conversations" element={<ConversationsPage />} />
          <Route path="/conversations/:conversationId" element={<ConversationDetailPage />} />
          <Route path="*" element={<Navigate to="/bots" replace />} />
        </Routes>
      </Layout>
    </AuthGate>
  )
}

import { Navigate, Route, Routes } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'
import BotsPage from './pages/BotsPage'
import BotDetailPage from './pages/BotDetailPage'
import KeysPage from './pages/KeysPage'
import ConversationsPage from './pages/ConversationsPage'
import ConversationDetailPage from './pages/ConversationDetailPage'

export default function App() {
  return (
    <AuthGate>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/bots" replace />} />
          <Route path="/bots" element={<BotsPage />} />
          <Route path="/bots/:botId" element={<BotDetailPage />} />
          <Route path="/keys" element={<KeysPage />} />
          <Route path="/conversations" element={<ConversationsPage />} />
          <Route path="/conversations/:conversationId" element={<ConversationDetailPage />} />
          <Route path="*" element={<Navigate to="/bots" replace />} />
        </Routes>
      </Layout>
    </AuthGate>
  )
}

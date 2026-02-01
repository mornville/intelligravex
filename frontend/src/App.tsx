import { Navigate, Outlet, RouterProvider, createBrowserRouter } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'
import BotsPage from './pages/BotsPage'
import BotDetailPage from './pages/BotDetailPage'
import KeysPage from './pages/KeysPage'
import ConversationsPage from './pages/ConversationsPage'
import ConversationDetailPage from './pages/ConversationDetailPage'
import GroupConversationPage from './pages/GroupConversationPage'
import DeveloperPage from './pages/DeveloperPage'
import LandingPage from './pages/LandingPage'

function AppShell() {
  return (
    <AuthGate>
      <Layout>
        <Outlet />
      </Layout>
    </AuthGate>
  )
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <LandingPage /> },
      { path: 'dashboard', element: <Navigate to="/bots" replace /> },
      { path: 'bots', element: <BotsPage /> },
      { path: 'bots/:botId', element: <BotDetailPage /> },
      { path: 'keys', element: <KeysPage /> },
      { path: 'developer', element: <DeveloperPage /> },
      { path: 'conversations', element: <ConversationsPage /> },
      { path: 'conversations/:conversationId', element: <ConversationDetailPage /> },
      { path: 'groups/:groupId', element: <GroupConversationPage /> },
      { path: '*', element: <Navigate to="/bots" replace /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}

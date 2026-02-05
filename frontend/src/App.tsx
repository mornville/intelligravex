import { Navigate, Outlet, RouterProvider, createBrowserRouter } from 'react-router-dom'
import AuthGate from './components/AuthGate'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import LandingPage from './pages/LandingPage'
import WidgetPage from './pages/WidgetPage'

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
  { path: '/widget', element: <WidgetPage /> },
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <LandingPage /> },
      { path: 'dashboard', element: <DashboardPage /> },
      { path: 'bots', element: <Navigate to="/dashboard" replace /> },
      { path: 'bots/:botId', element: <Navigate to="/dashboard" replace /> },
      { path: 'keys', element: <Navigate to="/dashboard" replace /> },
      { path: 'developer', element: <Navigate to="/dashboard" replace /> },
      { path: 'conversations', element: <Navigate to="/dashboard" replace /> },
      { path: 'conversations/:conversationId', element: <Navigate to="/dashboard" replace /> },
      { path: 'groups/:groupId', element: <Navigate to="/dashboard" replace /> },
      { path: '*', element: <Navigate to="/dashboard" replace /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}

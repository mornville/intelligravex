import Cocoa
import WebKit

@main
final class OverlayApp: NSObject, NSApplicationDelegate {
  private var window: NSWindow?
  private var webView: WKWebView?
  private var serverProcess: Process?
  private var outputBuffer = ""
  private var host = "127.0.0.1"
  private var port = "8000"

  func applicationDidFinishLaunching(_ notification: Notification) {
    NSApp.setActivationPolicy(.regular)
    setupWindow()
    startServer()
  }

  func applicationWillTerminate(_ notification: Notification) {
    serverProcess?.terminate()
  }

  private func setupWindow() {
    let size = NSSize(width: 200, height: 200)
    let rect = NSRect(origin: .zero, size: size)
    let window = NSWindow(contentRect: rect, styleMask: [.borderless], backing: .buffered, defer: false)
    window.isOpaque = false
    window.backgroundColor = .clear
    window.hasShadow = true
    window.level = .floating
    window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
    window.isMovableByWindowBackground = false
    positionWindow(window)

    let config = WKWebViewConfiguration()
    config.websiteDataStore = .default()
    let webView = WKWebView(frame: window.contentView?.bounds ?? rect, configuration: config)
    webView.autoresizingMask = [.width, .height]
    webView.setValue(false, forKey: "drawsBackground")
    window.contentView?.addSubview(webView)

    window.makeKeyAndOrderFront(nil)
    self.window = window
    self.webView = webView
    loadWidget()
  }

  private func positionWindow(_ window: NSWindow) {
    guard let screen = NSScreen.main else { return }
    let frame = screen.visibleFrame
    let x = frame.midX - window.frame.size.width / 2
    let y = frame.minY + 40
    window.setFrameOrigin(NSPoint(x: x, y: y))
  }

  private func startServer() {
    guard let serverURL = Bundle.main.url(forResource: "GravexServer", withExtension: nil) else {
      return
    }
    let process = Process()
    process.executableURL = serverURL
    var env = ProcessInfo.processInfo.environment
    env["VOICEBOT_OPEN_BROWSER"] = "0"
    env["VOICEBOT_OPEN_PATH"] = "/widget"
    env["PYTHONUNBUFFERED"] = "1"
    process.environment = env

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      if data.isEmpty { return }
      self?.handleOutput(data)
    }

    do {
      try process.run()
      serverProcess = process
    } catch {
      return
    }
  }

  private func handleOutput(_ data: Data) {
    guard let text = String(data: data, encoding: .utf8) else { return }
    outputBuffer.append(text)
    while let range = outputBuffer.range(of: "\n") {
      let line = outputBuffer[..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
      outputBuffer.removeSubrange(..<range.upperBound)
      if line.hasPrefix("GRAVEX_HOST=") {
        host = String(line.dropFirst("GRAVEX_HOST=".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        loadWidget()
      } else if line.hasPrefix("GRAVEX_PORT=") {
        port = String(line.dropFirst("GRAVEX_PORT=".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        loadWidget()
      }
    }
  }

  private func loadWidget() {
    guard let webView else { return }
    let urlString = "http://\(host):\(port)/widget"
    guard let url = URL(string: urlString) else { return }
    if webView.url?.absoluteString == urlString {
      return
    }
    let req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 5)
    webView.load(req)
  }
}

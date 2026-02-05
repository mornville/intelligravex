#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>
#import <Carbon/Carbon.h>
#import <CoreGraphics/CoreGraphics.h>

static EventHotKeyRef gHotKeyRefPrimary = NULL;
static EventHotKeyRef gHotKeyRefAlt = NULL;
static EventHandlerRef gHotKeyHandler = NULL;

static OSStatus HotKeyHandler(EventHandlerCallRef nextHandler, EventRef event, void *userData);
static void LogLine(NSString *line);

@interface DraggableWebView : WKWebView
@end

@implementation DraggableWebView
- (BOOL)mouseDownCanMoveWindow {
  return YES;
}
@end

@interface OverlayWindow : NSWindow
@end

@implementation OverlayWindow
- (BOOL)canBecomeKeyWindow {
  return YES;
}
- (BOOL)canBecomeMainWindow {
  return YES;
}
@end

@interface OverlayApp : NSObject <NSApplicationDelegate, WKNavigationDelegate>
@property (nonatomic, strong) NSWindow *window;
@property (nonatomic, strong) WKWebView *webView;
@property (nonatomic, strong) NSTask *serverProcess;
@property (nonatomic, strong) NSPipe *outputPipe;
@property (nonatomic, strong) NSMutableString *outputBuffer;
@property (nonatomic, copy) NSString *host;
@property (nonatomic, copy) NSString *port;
@property (nonatomic, strong) NSTimer *reloadTimer;
@end

@implementation OverlayApp

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  LogLine(@"App did finish launching");
  self.host = @"127.0.0.1";
  self.port = @"8000";
  self.outputBuffer = [NSMutableString string];
  [self setupWindow];
  [self startServer];
  LogLine(@"Registering hotkey");
  [self registerHotkey];
  LogLine(@"Hotkey registered");
}

- (void)applicationWillTerminate:(NSNotification *)notification {
  [self unregisterHotkey];
  if (self.serverProcess) {
    [self.serverProcess terminate];
  }
}

- (void)setupWindow {
  LogLine(@"Setting up window");
  @try {
    NSRect rect = NSMakeRect(0, 0, 200, 200);
    OverlayWindow *window = [[OverlayWindow alloc] initWithContentRect:rect
                                                   styleMask:NSWindowStyleMaskBorderless
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    LogLine(@"Window created");
    window.opaque = NO;
    window.backgroundColor = [NSColor colorWithCalibratedWhite:0 alpha:0.01];
    window.hasShadow = YES;
    window.movableByWindowBackground = YES;
    window.level = CGWindowLevelForKey(kCGScreenSaverWindowLevelKey);
    window.collectionBehavior =
      NSWindowCollectionBehaviorCanJoinAllSpaces |
      NSWindowCollectionBehaviorFullScreenAuxiliary |
      NSWindowCollectionBehaviorStationary;

    [self positionWindow:window];
    LogLine(@"Window positioned");

    WKWebViewConfiguration *config = [[WKWebViewConfiguration alloc] init];
    DraggableWebView *webView = [[DraggableWebView alloc] initWithFrame:window.contentView.bounds configuration:config];
    LogLine(@"WebView created");
    webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    @try {
      [webView setValue:@(NO) forKey:@"drawsBackground"];
    } @catch (NSException *ex) {
      LogLine([NSString stringWithFormat:@"KVC error: %@", ex.reason ?: @"unknown"]);
    }
    webView.navigationDelegate = self;
    [window.contentView addSubview:webView];
    LogLine(@"WebView added");

    [window makeKeyAndOrderFront:nil];
    self.window = window;
    self.webView = webView;
    [self loadLoadingScreen];
    LogLine(@"Window setup complete");
  } @catch (NSException *ex) {
    LogLine([NSString stringWithFormat:@"setupWindow exception: %@", ex.reason ?: @"unknown"]);
  }
}

- (void)positionWindow:(NSWindow *)window {
  NSScreen *screen = [NSScreen mainScreen];
  if (!screen) return;
  NSRect frame = screen.visibleFrame;
  CGFloat x = NSMidX(frame) - window.frame.size.width / 2.0;
  CGFloat y = NSMinY(frame) + 40.0;
  [window setFrameOrigin:NSMakePoint(x, y)];
}

- (void)startServer {
  NSString *serverPath = [[NSBundle mainBundle] pathForResource:@"GravexServer" ofType:nil];
  if (!serverPath) {
    LogLine(@"GravexServer binary not found in bundle");
    return;
  }
  BOOL isExecutable = [[NSFileManager defaultManager] isExecutableFileAtPath:serverPath];
  LogLine([NSString stringWithFormat:@"GravexServer executable=%@", isExecutable ? @"YES" : @"NO"]);
  LogLine([NSString stringWithFormat:@"Launching server at %@", serverPath]);

  NSTask *task = [[NSTask alloc] init];
  task.executableURL = [NSURL fileURLWithPath:serverPath];

  NSMutableDictionary *env = [NSMutableDictionary dictionaryWithDictionary:[[NSProcessInfo processInfo] environment]];
  env[@"VOICEBOT_OPEN_BROWSER"] = @"0";
  env[@"VOICEBOT_OPEN_PATH"] = @"/widget";
  env[@"VOICEBOT_PROMPT_INITIAL_PORT"] = @"0";
  env[@"PYTHONUNBUFFERED"] = @"1";
  task.environment = env;

  NSPipe *pipe = [NSPipe pipe];
  task.standardOutput = pipe;
  task.standardError = pipe;
  self.outputPipe = pipe;

  __weak typeof(self) weakSelf = self;
  pipe.fileHandleForReading.readabilityHandler = ^(NSFileHandle *handle) {
    NSData *data = handle.availableData;
    if (data.length == 0) return;
    [weakSelf handleOutputData:data];
  };

  NSError *error = nil;
  if (![task launchAndReturnError:&error]) {
    LogLine([NSString stringWithFormat:@"Failed to launch server: %@", error ? error.localizedDescription : @"unknown error"]);
    return;
  }
  self.serverProcess = task;
}

- (void)handleOutputData:(NSData *)data {
  NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
  if (!text) return;
  [self.outputBuffer appendString:text];
  NSRange range;
  while ((range = [self.outputBuffer rangeOfString:@"\n"]).location != NSNotFound) {
    NSString *line = [[self.outputBuffer substringToIndex:range.location] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    [self.outputBuffer deleteCharactersInRange:NSMakeRange(0, range.location + 1)];
    if (line.length > 0) {
      LogLine([NSString stringWithFormat:@"server: %@", line]);
    }
    if ([line hasPrefix:@"GRAVEX_HOST="]) {
      self.host = [[line substringFromIndex:[@"GRAVEX_HOST=" length]] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
      [self loadWidget];
    } else if ([line hasPrefix:@"GRAVEX_PORT="]) {
      self.port = [[line substringFromIndex:[@"GRAVEX_PORT=" length]] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
      [self loadWidget];
    }
  }
}

- (void)loadWidget {
  [self loadWidgetForce:NO];
}

- (void)loadLoadingScreen {
  if (!self.webView) return;
  NSString *html =
    @"<!doctype html>"
    "<html><head><meta charset=\"utf-8\"/>"
    "<style>"
    "html,body{margin:0;height:100%;font-family:-apple-system,Helvetica,Arial,sans-serif;background:transparent;color:#e5e7eb;}"
    ".wrap{height:100%;display:flex;align-items:center;justify-content:center;}"
    ".card{width:160px;height:160px;border-radius:22px;background:rgba(8,12,28,0.85);"
    "box-shadow:0 18px 36px rgba(2,6,23,0.6);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;}"
    ".spinner{width:28px;height:28px;border-radius:50%;border:3px solid rgba(255,255,255,0.15);"
    "border-top-color:rgba(138,180,248,0.95);animation:spin 1s linear infinite;}"
    ".label{font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:rgba(203,213,225,0.85);}"
    "@keyframes spin{to{transform:rotate(360deg);}}"
    "</style></head>"
    "<body><div class=\"wrap\"><div class=\"card\"><div class=\"spinner\"></div>"
    "<div class=\"label\">Starting...</div></div></div></body></html>";
  dispatch_async(dispatch_get_main_queue(), ^{
    [self.webView loadHTMLString:html baseURL:nil];
  });
}

- (void)loadWidgetForce:(BOOL)force {
  if (!self.webView) return;
  NSString *urlString = [NSString stringWithFormat:@"http://%@:%@/widget", self.host, self.port];
  NSURL *url = [NSURL URLWithString:urlString];
  if (!url) return;
  if (!force && self.webView.URL && [self.webView.URL.absoluteString isEqualToString:urlString]) return;
  NSURLRequest *req = [NSURLRequest requestWithURL:url cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:5];
  dispatch_async(dispatch_get_main_queue(), ^{
    [self.webView loadRequest:req];
  });
}

- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(WKNavigation *)navigation withError:(NSError *)error {
  [self loadLoadingScreen];
  [self scheduleReload];
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation withError:(NSError *)error {
  [self loadLoadingScreen];
  [self scheduleReload];
}

- (void)scheduleReload {
  if (self.reloadTimer) return;
  self.reloadTimer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                      target:self
                                                    selector:@selector(retryLoad)
                                                    userInfo:nil
                                                     repeats:NO];
}

- (void)retryLoad {
  self.reloadTimer = nil;
  [self loadWidgetForce:YES];
}

- (void)toggleWindow {
  if (!self.window) return;
  if (self.window.isVisible) {
    [self.window orderOut:nil];
    return;
  }
  [self positionWindow:self.window];
  [self.window makeKeyAndOrderFront:nil];
  [NSApp activateIgnoringOtherApps:YES];
}

- (void)registerHotkey {
  EventTypeSpec eventType;
  eventType.eventClass = kEventClassKeyboard;
  eventType.eventKind = kEventHotKeyPressed;
  if (!gHotKeyHandler) {
    InstallEventHandler(GetApplicationEventTarget(), HotKeyHandler, 1, &eventType, (__bridge void *)self, &gHotKeyHandler);
  }
  UInt32 keyCode = kVK_ANSI_M;
  EventHotKeyID hotKeyID;
  hotKeyID.signature = 'grvx';

  hotKeyID.id = 1;
  UInt32 modifiers = cmdKey | shiftKey;
  OSStatus status = RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &gHotKeyRefPrimary);
  if (status != noErr) {
    NSLog(@"Hotkey Cmd+Shift+M registration failed: %d", (int)status);
  }

  hotKeyID.id = 2;
  UInt32 altModifiers = cmdKey | optionKey | controlKey;
  status = RegisterEventHotKey(keyCode, altModifiers, hotKeyID, GetApplicationEventTarget(), 0, &gHotKeyRefAlt);
  if (status != noErr) {
    NSLog(@"Hotkey Ctrl+Option+Cmd+M registration failed: %d", (int)status);
  }
}

- (void)unregisterHotkey {
  if (gHotKeyRefPrimary) {
    UnregisterEventHotKey(gHotKeyRefPrimary);
    gHotKeyRefPrimary = NULL;
  }
  if (gHotKeyRefAlt) {
    UnregisterEventHotKey(gHotKeyRefAlt);
    gHotKeyRefAlt = NULL;
  }
  if (gHotKeyHandler) {
    RemoveEventHandler(gHotKeyHandler);
    gHotKeyHandler = NULL;
  }
}

@end

static OSStatus HotKeyHandler(EventHandlerCallRef nextHandler, EventRef event, void *userData) {
  OverlayApp *app = (__bridge OverlayApp *)userData;
  [app toggleWindow];
  return noErr;
}

static void LogLine(NSString *line) {
  NSString *logDir = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Logs"];
  NSString *path = [logDir stringByAppendingPathComponent:@"GravexOverlay.log"];
  [[NSFileManager defaultManager] createDirectoryAtPath:logDir
                            withIntermediateDirectories:YES
                                             attributes:nil
                                                  error:nil];
  NSString *msg = [NSString stringWithFormat:@"%@ %@\n", [NSDate date], line ?: @"(null)"];
  NSData *data = [msg dataUsingEncoding:NSUTF8StringEncoding];
  NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:path];
  if (!handle) {
    [[NSFileManager defaultManager] createFileAtPath:path contents:nil attributes:nil];
    handle = [NSFileHandle fileHandleForWritingAtPath:path];
  }
  if (!handle) return;
  @try {
    [handle seekToEndOfFile];
    [handle writeData:data];
  } @catch (__unused NSException *ex) {
  }
  [handle closeFile];
}

int main(int argc, const char * argv[]) {
  @autoreleasepool {
    NSApplication *app = [NSApplication sharedApplication];
    OverlayApp *delegate = [[OverlayApp alloc] init];
    app.delegate = delegate;
    [app run];
  }
  return 0;
}

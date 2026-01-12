package main

import (
	"crypto/rand"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
)

type Config struct {
	SourceRepo        string `json:"source_repo"`
	DestDir           string `json:"dest_dir"`
	DestBase          string `json:"dest_base"`
	RemoteURL         string `json:"remote_url"`
	Branch            string `json:"branch"`
	RecurseSubmodules *bool  `json:"recurse_submodules"`
	UseLocal          *bool  `json:"use_local"`
}

type Registry struct {
	Version int                 `json:"version"`
	Items   map[string]SplitRef `json:"items"`
}

type SplitRef struct {
	ID         string    `json:"id"`
	Name       string    `json:"name"`
	Path       string    `json:"path"`
	SourceRepo string    `json:"source_repo"`
	Branch     string    `json:"branch,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
}

const (
	defaultSourceRepo = "/Users/elidoruiz/dev/repos/gh-candorverse"
	defaultDestBase   = "splits"
	defaultImage      = "repo-split-codex:latest"
	registryFileName  = ".splits.json"
	idLength          = 3
)

//go:embed Dockerfile.codex
var embeddedDockerfile string

type cloneOptions struct {
	src               string
	dest              string
	destBase          string
	name              string
	remote            string
	branch            string
	recurseSubmodules bool
	local             bool
	dryRun            bool
}

type listOptions struct {
	destBase string
}

type removeOptions struct {
	destBase string
	name     string
	delete   bool
}

type pruneOptions struct {
	destBase string
	delete   bool
	dryRun   bool
}

type containerOptions struct {
	destBase       string
	id             string
	image          string
	dockerfile     string
	buildImage     bool
	containerName  string
	workspace      string
	hostNetwork    bool
	passEnv        bool
	mountAws       bool
	mountPgpass    bool
	mountCodex     bool
	mountCodexHome bool
	codexPath      string
	installNode    bool
	installCodex   bool
	resume         string
	shell          bool
	stop           bool
	dryRun         bool
}

type codexOptions struct {
	destBase string
	id       string
	resume   string
}

type resumeOptions struct {
	destBase string
	id       string
	session  string
}

type restartOptions struct {
	destBase       string
	image          string
	dockerfile     string
	rebuildImage   bool
	hostNetwork    bool
	passEnv        bool
	mountAws       bool
	mountPgpass    bool
	mountCodex     bool
	mountCodexHome bool
	codexPath      string
	workspace      string
	dryRun         bool
}

type authSyncOptions struct {
	codexHome string
}

var (
	configPath string

	rootCloneOpts cloneOptions
	cloneOpts     cloneOptions
	listOpts      listOptions
	removeOpts    removeOptions
	pruneOpts     pruneOptions
	containerOpts containerOptions
	codexOpts     codexOptions
	resumeOpts    resumeOptions
	restartOpts   restartOptions
	authSyncOpts  authSyncOptions
)

var rootCmd = &cobra.Command{
	Use:          "repo-split",
	Short:        "Clone and manage local repo splits for Codex",
	SilenceUsage: true,
	Run: func(cmd *cobra.Command, args []string) {
		runClone(cmd, &rootCloneOpts, args)
	},
}

func main() {
	if err := Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.PersistentFlags().StringVar(&configPath, "config", "config.json", "path to config JSON file")
	addCloneFlags(rootCmd, &rootCloneOpts)

	rootCmd.AddCommand(
		newCloneCmd(),
		newListCmd(),
		newRemoveCmd(),
		newPruneCmd(),
		newContainerCmd(),
		newCodexCmd(),
		newResumeCmd(),
		newRestartCmd(),
		newAuthSyncCmd(),
	)
}

func addCloneFlags(cmd *cobra.Command, opts *cloneOptions) {
	cmd.Flags().StringVar(&opts.src, "src", "", "path to local git repo (overrides config)")
	cmd.Flags().StringVar(&opts.dest, "dest", "", "destination directory (overrides config)")
	cmd.Flags().StringVar(&opts.destBase, "dest-base", "", "base directory to create a new clone when -dest is not set")
	cmd.Flags().StringVar(&opts.name, "name", "", "nickname for the clone (used for list/remove)")
	cmd.Flags().StringVar(&opts.remote, "remote", "", "remote URL to set as origin after clone")
	cmd.Flags().StringVar(&opts.branch, "branch", "", "git branch to clone (overrides config)")
	cmd.Flags().BoolVar(&opts.recurseSubmodules, "recurse-submodules", false, "clone submodules (overrides config if set)")
	cmd.Flags().BoolVar(&opts.local, "local", false, "use --local when cloning (overrides config if set)")
	cmd.Flags().BoolVar(&opts.dryRun, "dry-run", false, "print the git command without running it")
}

func newCloneCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "clone",
		Short:        "Create a split (default command)",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runClone(cmd, &cloneOpts, args)
		},
	}
	addCloneFlags(cmd, &cloneOpts)
	return cmd
}

func newListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "list",
		Short:        "List splits with IDs, age, and last update",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runList(cmd, &listOpts, args)
		},
	}
	cmd.Flags().StringVar(&listOpts.destBase, "dest-base", "", "base directory containing splits")
	return cmd
}

func newRemoveCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "remove",
		Short:        "Remove a split by name",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runRemove(cmd, &removeOpts, args)
		},
	}
	cmd.Flags().StringVar(&removeOpts.destBase, "dest-base", "", "base directory containing splits")
	cmd.Flags().StringVar(&removeOpts.name, "name", "", "nickname to remove")
	cmd.Flags().BoolVar(&removeOpts.delete, "delete", true, "delete the repo directory as well as registry entry")
	return cmd
}

func newPruneCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "prune",
		Short:        "Remove all splits",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runPrune(cmd, &pruneOpts, args)
		},
	}
	cmd.Flags().StringVar(&pruneOpts.destBase, "dest-base", "", "base directory containing splits")
	cmd.Flags().BoolVar(&pruneOpts.delete, "delete", true, "delete the repo directories as well as registry entries")
	cmd.Flags().BoolVar(&pruneOpts.dryRun, "dry-run", false, "print what would be removed without deleting anything")
	return cmd
}

func newContainerCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "container",
		Short:        "Run codex inside a Docker container for a split",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runContainer(cmd, &containerOpts, args)
		},
	}
	cmd.Flags().StringVar(&containerOpts.destBase, "dest-base", "", "base directory containing splits")
	cmd.Flags().StringVar(&containerOpts.id, "id", "", "id of the split to open")
	cmd.Flags().StringVar(&containerOpts.image, "image", defaultImage, "docker image to use")
	cmd.Flags().StringVar(&containerOpts.dockerfile, "dockerfile", "", "path to Dockerfile to build the image if missing")
	cmd.Flags().BoolVar(&containerOpts.buildImage, "build-image", true, "build the image if it is missing")
	cmd.Flags().StringVar(&containerOpts.containerName, "container-name", "", "override docker container name")
	cmd.Flags().StringVar(&containerOpts.workspace, "workspace", "/workspace", "container workspace path")
	cmd.Flags().BoolVar(&containerOpts.hostNetwork, "host-network", true, "use host networking")
	cmd.Flags().BoolVar(&containerOpts.passEnv, "pass-env", true, "pass host environment variables into the container")
	cmd.Flags().BoolVar(&containerOpts.mountAws, "mount-aws", true, "mount host ~/.aws into the container")
	cmd.Flags().BoolVar(&containerOpts.mountPgpass, "mount-pgpass", true, "mount host ~/.pgpass into the container")
	cmd.Flags().BoolVar(&containerOpts.mountCodex, "mount-codex", false, "mount host codex binary into the container")
	cmd.Flags().BoolVar(&containerOpts.mountCodexHome, "mount-codex-home", true, "mount host ~/.codex into the container")
	cmd.Flags().StringVar(&containerOpts.codexPath, "codex-path", "", "path to host codex binary (optional)")
	cmd.Flags().BoolVar(&containerOpts.installNode, "install-node", true, "install Node.js in the container if missing")
	cmd.Flags().BoolVar(&containerOpts.installCodex, "install-codex", true, "install Codex CLI in the container if missing (npm i -g @openai/codex)")
	cmd.Flags().StringVar(&containerOpts.resume, "resume", "", "resume a codex session id inside the container")
	cmd.Flags().BoolVar(&containerOpts.shell, "shell", false, "open a shell in the container instead of running codex")
	cmd.Flags().BoolVar(&containerOpts.stop, "stop", false, "stop and remove the container")
	cmd.Flags().BoolVar(&containerOpts.dryRun, "dry-run", false, "print docker commands without running them")
	return cmd
}

func newCodexCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "codex",
		Short:        "Start a Codex session in a split by ID",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runCodex(cmd, &codexOpts, args)
		},
	}
	cmd.Flags().StringVar(&codexOpts.destBase, "dest-base", "", "base directory containing splits")
	cmd.Flags().StringVar(&codexOpts.id, "id", "", "id of the split to open")
	cmd.Flags().StringVar(&codexOpts.resume, "resume", "", "resume a codex session id")
	return cmd
}

func newResumeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "resume",
		Short:        "Resume a Codex session in a split by ID",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runResume(cmd, &resumeOpts, args)
		},
	}
	cmd.Flags().StringVar(&resumeOpts.destBase, "dest-base", "", "base directory containing splits")
	cmd.Flags().StringVar(&resumeOpts.id, "id", "", "id of the split to open")
	cmd.Flags().StringVar(&resumeOpts.session, "session", "", "codex session id to resume")
	return cmd
}

func newRestartCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "restart",
		Short:        "Restart all repo-split containers (rebuilds image by default)",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runRestart(cmd, &restartOpts, args)
		},
	}
	cmd.Flags().StringVar(&restartOpts.destBase, "dest-base", "", "base directory containing splits")
	cmd.Flags().StringVar(&restartOpts.image, "image", defaultImage, "docker image to use")
	cmd.Flags().StringVar(&restartOpts.dockerfile, "dockerfile", "", "path to Dockerfile to build the image")
	cmd.Flags().BoolVar(&restartOpts.rebuildImage, "rebuild-image", true, "rebuild the image before restarting containers")
	cmd.Flags().BoolVar(&restartOpts.hostNetwork, "host-network", true, "use host networking")
	cmd.Flags().BoolVar(&restartOpts.passEnv, "pass-env", true, "pass host environment variables into the container")
	cmd.Flags().BoolVar(&restartOpts.mountAws, "mount-aws", true, "mount host ~/.aws into the container")
	cmd.Flags().BoolVar(&restartOpts.mountPgpass, "mount-pgpass", true, "mount host ~/.pgpass into the container")
	cmd.Flags().BoolVar(&restartOpts.mountCodex, "mount-codex", false, "mount host codex binary into the container")
	cmd.Flags().BoolVar(&restartOpts.mountCodexHome, "mount-codex-home", true, "mount host ~/.codex into the container")
	cmd.Flags().StringVar(&restartOpts.codexPath, "codex-path", "", "path to host codex binary (optional)")
	cmd.Flags().StringVar(&restartOpts.workspace, "workspace", "/workspace", "container workspace path")
	cmd.Flags().BoolVar(&restartOpts.dryRun, "dry-run", false, "print docker commands without running them")
	return cmd
}

func newAuthSyncCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "auth-sync",
		Short:        "Ensure Codex uses file-based auth and report cache status",
		SilenceUsage: true,
		Run: func(cmd *cobra.Command, args []string) {
			runAuthSync(cmd, &authSyncOpts, args)
		},
	}
	cmd.Flags().StringVar(&authSyncOpts.codexHome, "codex-home", "", "override codex home directory (default: ~/.codex)")
	return cmd
}

func runClone(cmd *cobra.Command, opts *cloneOptions, args []string) {
	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	src := firstNonEmpty(opts.src, cfg.SourceRepo, defaultSourceRepo)
	dest := firstNonEmpty(opts.dest, cfg.DestDir)
	destBase := firstNonEmpty(opts.destBase, cfg.DestBase)
	branch := firstNonEmpty(opts.branch, cfg.Branch)
	name := strings.TrimSpace(opts.name)
	remoteURL := firstNonEmpty(opts.remote, cfg.RemoteURL)

	recurse := valueOrDefault(cfg.RecurseSubmodules, false)
	if cmd.Flags().Changed("recurse-submodules") {
		recurse = opts.recurseSubmodules
	}

	useLocal := valueOrDefault(cfg.UseLocal, false)
	if cmd.Flags().Changed("local") {
		useLocal = opts.local
	}

	src = resolvePath(src)
	if src == "" {
		exitErr(errors.New("source repo is required"))
	}

	destBase = resolvePath(destBase)
	if dest == "" {
		if destBase == "" {
			destBase = resolvePath(defaultDestBase)
		}
		if name != "" {
			dest = filepath.Join(destBase, name)
		} else {
			dest = defaultDestPath(src, destBase)
		}
	}
	dest = resolvePath(dest)
	if destBase == "" {
		destBase = resolvePath(filepath.Dir(dest))
	}

	if err := validateSourceRepo(src); err != nil {
		exitErr(err)
	}
	if err := validateDest(dest); err != nil {
		exitErr(err)
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}
	if reg.Items == nil {
		reg.Items = make(map[string]SplitRef)
	}
	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}

	if name == "" {
		name = repoBaseName(dest)
	}
	if _, exists := reg.Items[name]; exists {
		exitErr(fmt.Errorf("name already exists: %s", name))
	}

	id := generateID(reg)

	cloneArgs := []string{"clone"}
	if useLocal {
		cloneArgs = append(cloneArgs, "--local")
	}
	if recurse {
		cloneArgs = append(cloneArgs, "--recurse-submodules")
	}
	if branch != "" {
		cloneArgs = append(cloneArgs, "--branch", branch)
	}
	cloneArgs = append(cloneArgs, src, dest)

	cmdLine := "git " + strings.Join(cloneArgs, " ")
	if opts.dryRun {
		fmt.Println(cmdLine)
		fmt.Printf("Path: %s\n", dest)
		return
	}

	gitCmd := exec.Command("git", cloneArgs...)
	gitCmd.Stdout = os.Stdout
	gitCmd.Stderr = os.Stderr
	if err := gitCmd.Run(); err != nil {
		exitErr(fmt.Errorf("git clone failed: %w", err))
	}

	if err := ensureRemoteURL(dest, src, remoteURL); err != nil {
		fmt.Fprintln(os.Stderr, "warning:", err)
	}

	reg.Items[name] = SplitRef{
		ID:         id,
		Name:       name,
		Path:       dest,
		SourceRepo: src,
		Branch:     branch,
		CreatedAt:  time.Now(),
	}
	if err := saveRegistry(regPath, reg); err != nil {
		exitErr(err)
	}

	fmt.Printf("ID: %s\n", id)
	fmt.Printf("Name: %s\n", name)
	fmt.Printf("Path: %s\n", dest)
}

func runList(cmd *cobra.Command, opts *listOptions, args []string) {
	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}
	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}
	if len(reg.Items) == 0 {
		fmt.Printf("No splits found in %s\n", destBase)
		return
	}

	names := make([]string, 0, len(reg.Items))
	for name := range reg.Items {
		names = append(names, name)
	}
	sort.Strings(names)

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tNAME\tAGE\tUPDATED\tPATH")
	for _, name := range names {
		entry := reg.Items[name]
		age := "unknown"
		if !entry.CreatedAt.IsZero() {
			age = humanDuration(time.Since(entry.CreatedAt))
		}

		updated := "unknown"
		if pathExists(entry.Path) && looksLikeGitRepo(entry.Path) {
			if t, err := gitLastUpdated(entry.Path); err == nil {
				updated = t.Local().Format("2006-01-02 15:04")
			}
		} else if !pathExists(entry.Path) {
			updated = "missing"
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", entry.ID, name, age, updated, entry.Path)
	}
	_ = w.Flush()
}

func runRemove(cmd *cobra.Command, opts *removeOptions, args []string) {
	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	name := strings.TrimSpace(opts.name)
	if name == "" {
		exitErr(errors.New("name is required for remove"))
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}
	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}
	entry, ok := reg.Items[name]
	if !ok {
		exitErr(fmt.Errorf("name not found: %s", name))
	}

	if opts.delete {
		if err := os.RemoveAll(entry.Path); err != nil {
			exitErr(fmt.Errorf("failed to remove path: %w", err))
		}
	}

	delete(reg.Items, name)
	if err := saveRegistry(regPath, reg); err != nil {
		exitErr(err)
	}

	fmt.Printf("Removed %s\n", name)
}

func runPrune(cmd *cobra.Command, opts *pruneOptions, args []string) {
	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}

	if len(reg.Items) == 0 {
		fmt.Printf("No splits found in %s\n", destBase)
		return
	}

	if opts.dryRun {
		fmt.Printf("Would prune %d split(s):\n", len(reg.Items))
		names := make([]string, 0, len(reg.Items))
		for name := range reg.Items {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			entry := reg.Items[name]
			fmt.Printf("- %s (%s)\n", name, entry.Path)
		}
		return
	}

	if opts.delete {
		seen := make(map[string]struct{})
		for _, entry := range reg.Items {
			if entry.Path == "" {
				continue
			}
			if _, ok := seen[entry.Path]; ok {
				continue
			}
			seen[entry.Path] = struct{}{}
			if err := os.RemoveAll(entry.Path); err != nil {
				exitErr(fmt.Errorf("failed to remove path: %w", err))
			}
		}
	}

	removed := len(reg.Items)
	reg.Items = make(map[string]SplitRef)
	if err := saveRegistry(regPath, reg); err != nil {
		exitErr(err)
	}

	fmt.Printf("Pruned %d split(s)\n", removed)
}

func runContainer(cmd *cobra.Command, opts *containerOptions, args []string) {
	id := strings.TrimSpace(opts.id)
	if id == "" {
		exitErr(errors.New("id is required for container"))
	}
	buildImage := opts.buildImage
	if !cmd.Flags().Changed("build-image") && opts.image != defaultImage {
		buildImage = false
	}
	if opts.shell && opts.stop {
		exitErr(errors.New("choose only one of -shell or -stop"))
	}
	if opts.shell && strings.TrimSpace(opts.resume) != "" {
		exitErr(errors.New("choose only one of -shell or -resume"))
	}
	if opts.stop && strings.TrimSpace(opts.resume) != "" {
		exitErr(errors.New("choose only one of -stop or -resume"))
	}
	if opts.mountCodex && opts.installCodex {
		exitErr(errors.New("choose only one of -mount-codex or -install-codex"))
	}

	if err := ensureDocker(); err != nil {
		exitErr(err)
	}

	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}
	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}

	entry, ok := findByID(reg, id)
	if !ok {
		exitErr(fmt.Errorf("id not found: %s", id))
	}

	repoPath := resolvePath(entry.Path)
	if !pathExists(repoPath) {
		exitErr(fmt.Errorf("path does not exist: %s", repoPath))
	}

	containerName := strings.TrimSpace(opts.containerName)
	if containerName == "" {
		containerName = fmt.Sprintf("repo-split-%s", id)
	}

	envFile := ""
	cleanupEnv := func() {}
	if opts.passEnv {
		filePath, cleanup, err := writeEnvFile(os.Environ(), containerEnvOverrides(), containerEnvDenylist())
		if err != nil {
			exitErr(err)
		}
		envFile = filePath
		cleanupEnv = cleanup
	}
	defer cleanupEnv()

	awsPath := ""
	if opts.mountAws {
		if home, err := os.UserHomeDir(); err == nil {
			path := filepath.Join(home, ".aws")
			if pathExists(path) {
				awsPath = path
			}
		}
	}
	pgpassPath := ""
	if opts.mountPgpass {
		if home, err := os.UserHomeDir(); err == nil {
			path := filepath.Join(home, ".pgpass")
			if pathExists(path) {
				pgpassPath = path
			}
		}
	}

	if !opts.stop {
		if exists, _, err := containerState(containerName); err != nil {
			exitErr(err)
		} else if !exists {
			if err := ensureDockerImage(opts.image, opts.dockerfile, embeddedDockerfile, buildImage, false, opts.dryRun); err != nil {
				exitErr(err)
			}
		}
	}

	if opts.stop {
		if opts.dryRun {
			fmt.Printf("docker rm -f %s\n", containerName)
			return
		}
		if exists, _, _ := containerState(containerName); !exists {
			fmt.Printf("Container not found: %s\n", containerName)
			return
		}
		rmCmd := exec.Command("docker", "rm", "-f", containerName)
		rmCmd.Stdout = os.Stdout
		rmCmd.Stderr = os.Stderr
		if err := rmCmd.Run(); err != nil {
			exitErr(fmt.Errorf("docker rm failed: %w", err))
		}
		fmt.Printf("Removed container %s\n", containerName)
		return
	}

	codexPath := strings.TrimSpace(opts.codexPath)
	if codexPath == "" && opts.mountCodex {
		if p, err := exec.LookPath("codex"); err == nil {
			codexPath = p
		} else {
			exitErr(errors.New("codex binary not found on host; install codex or pass -mount-codex=false"))
		}
	}

	if err := ensureContainerRunning(containerName, opts.image, repoPath, opts.workspace, opts.hostNetwork, opts.mountCodex, opts.mountCodexHome, codexPath, envFile, awsPath, pgpassPath, opts.dryRun); err != nil {
		exitErr(err)
	}

	if opts.dryRun {
		return
	}

	if opts.installCodex || opts.mountCodex {
		if err := ensureNodeInContainer(containerName, opts.installNode); err != nil {
			exitErr(err)
		}
	}
	if opts.installCodex {
		if err := ensureCodexInContainer(containerName); err != nil {
			exitErr(err)
		}
	} else if !containerHasCommand(containerName, "codex") {
		exitErr(errors.New("codex not found in container; re-run with -install-codex or use an image with codex"))
	}

	if opts.shell {
		shellCmd := exec.Command("docker", "exec", "-it", containerName, "bash")
		shellCmd.Stdout = os.Stdout
		shellCmd.Stderr = os.Stderr
		shellCmd.Stdin = os.Stdin
		if err := shellCmd.Run(); err != nil {
			exitErr(fmt.Errorf("docker exec shell failed: %w", err))
		}
		return
	}

	codexArgs := args
	resumeID := strings.TrimSpace(opts.resume)
	execArgs := []string{"exec", "-it", containerName, "codex"}
	if resumeID != "" {
		execArgs = append(execArgs, "resume", resumeID)
	}
	execArgs = append(execArgs, codexArgs...)
	execCmd := exec.Command("docker", execArgs...)
	execCmd.Stdout = os.Stdout
	execCmd.Stderr = os.Stderr
	execCmd.Stdin = os.Stdin
	if err := execCmd.Run(); err != nil {
		exitErr(fmt.Errorf("docker exec codex failed: %w", err))
	}
}

func runCodex(cmd *cobra.Command, opts *codexOptions, args []string) {
	id := strings.TrimSpace(opts.id)
	if id == "" {
		exitErr(errors.New("id is required for codex"))
	}

	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}

	entry, ok := findByID(reg, id)
	if !ok {
		exitErr(fmt.Errorf("id not found: %s", id))
	}

	if !pathExists(entry.Path) {
		exitErr(fmt.Errorf("path does not exist: %s", entry.Path))
	}

	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}

	codexArgs := args
	resumeID := strings.TrimSpace(opts.resume)
	if resumeID != "" {
		codexArgs = append([]string{"resume", resumeID}, codexArgs...)
	}
	execCmd := exec.Command("codex", codexArgs...)
	execCmd.Stdout = os.Stdout
	execCmd.Stderr = os.Stderr
	execCmd.Stdin = os.Stdin
	execCmd.Dir = entry.Path
	if err := execCmd.Run(); err != nil {
		exitErr(fmt.Errorf("codex failed: %w", err))
	}
}

func runResume(cmd *cobra.Command, opts *resumeOptions, args []string) {
	sessionID := strings.TrimSpace(opts.session)
	if sessionID == "" {
		exitErr(errors.New("session is required for resume"))
	}
	id := strings.TrimSpace(opts.id)
	if id == "" {
		exitErr(errors.New("id is required for resume"))
	}

	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}
	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}

	entry, ok := findByID(reg, id)
	if !ok {
		exitErr(fmt.Errorf("id not found: %s", id))
	}

	if !pathExists(entry.Path) {
		exitErr(fmt.Errorf("path does not exist: %s", entry.Path))
	}

	codexArgs := append([]string{"resume", sessionID}, args...)
	execCmd := exec.Command("codex", codexArgs...)
	execCmd.Stdout = os.Stdout
	execCmd.Stderr = os.Stderr
	execCmd.Stdin = os.Stdin
	execCmd.Dir = entry.Path
	if err := execCmd.Run(); err != nil {
		exitErr(fmt.Errorf("codex resume failed: %w", err))
	}
}

func runRestart(cmd *cobra.Command, opts *restartOptions, args []string) {
	if err := ensureDocker(); err != nil {
		exitErr(err)
	}

	cfg, _, cfgErr := loadConfig(configPath)
	if cfgErr != nil {
		exitErr(cfgErr)
	}

	destBase := firstNonEmpty(opts.destBase, cfg.DestBase, defaultDestBase)
	destBase = resolvePath(destBase)
	if destBase == "" {
		exitErr(errors.New("dest base is required"))
	}

	regPath := registryPath(destBase)
	reg, _, err := loadRegistry(regPath)
	if err != nil {
		exitErr(err)
	}
	if ensureRegistryIDs(&reg) {
		if err := saveRegistry(regPath, reg); err != nil {
			exitErr(err)
		}
	}
	if len(reg.Items) == 0 {
		fmt.Printf("No splits found in %s\n", destBase)
		return
	}

	buildImage := opts.rebuildImage
	if !cmd.Flags().Changed("rebuild-image") && opts.image != defaultImage {
		buildImage = false
	}
	if err := ensureDockerImage(opts.image, opts.dockerfile, embeddedDockerfile, true, buildImage, opts.dryRun); err != nil {
		exitErr(err)
	}

	envFile := ""
	cleanupEnv := func() {}
	if opts.passEnv {
		filePath, cleanup, err := writeEnvFile(os.Environ(), containerEnvOverrides(), containerEnvDenylist())
		if err != nil {
			exitErr(err)
		}
		envFile = filePath
		cleanupEnv = cleanup
	}
	defer cleanupEnv()

	awsPath := ""
	if opts.mountAws {
		if home, err := os.UserHomeDir(); err == nil {
			path := filepath.Join(home, ".aws")
			if pathExists(path) {
				awsPath = path
			}
		}
	}
	pgpassPath := ""
	if opts.mountPgpass {
		if home, err := os.UserHomeDir(); err == nil {
			path := filepath.Join(home, ".pgpass")
			if pathExists(path) {
				pgpassPath = path
			}
		}
	}

	codexPath := strings.TrimSpace(opts.codexPath)
	if codexPath == "" && opts.mountCodex {
		if p, err := exec.LookPath("codex"); err == nil {
			codexPath = p
		} else {
			exitErr(errors.New("codex binary not found on host; install codex or pass -mount-codex=false"))
		}
	}

	names := make([]string, 0, len(reg.Items))
	for name := range reg.Items {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		entry := reg.Items[name]
		containerName := fmt.Sprintf("repo-split-%s", entry.ID)
		repoPath := resolvePath(entry.Path)
		if !pathExists(repoPath) {
			fmt.Fprintf(os.Stderr, "warning: path missing for %s (%s)\n", name, repoPath)
			continue
		}

		if exists, _, err := containerState(containerName); err != nil {
			exitErr(err)
		} else if exists {
			if opts.dryRun {
				fmt.Printf("docker rm -f %s\n", containerName)
			} else {
				rmCmd := exec.Command("docker", "rm", "-f", containerName)
				rmCmd.Stdout = os.Stdout
				rmCmd.Stderr = os.Stderr
				if err := rmCmd.Run(); err != nil {
					exitErr(fmt.Errorf("docker rm failed: %w", err))
				}
			}
		}

		if err := ensureContainerRunning(containerName, opts.image, repoPath, opts.workspace, opts.hostNetwork, opts.mountCodex, opts.mountCodexHome, codexPath, envFile, awsPath, pgpassPath, opts.dryRun); err != nil {
			exitErr(err)
		}
	}

	fmt.Printf("Restarted %d container(s)\n", len(reg.Items))
}

func runAuthSync(cmd *cobra.Command, opts *authSyncOptions, args []string) {
	codexHome := strings.TrimSpace(opts.codexHome)
	if codexHome == "" {
		if home, err := os.UserHomeDir(); err == nil {
			codexHome = filepath.Join(home, ".codex")
		}
	}
	if codexHome == "" {
		exitErr(errors.New("unable to determine codex home"))
	}
	codexHome = resolvePath(codexHome)

	changed, err := ensureCodexAuthFileStore(codexHome)
	if err != nil {
		exitErr(err)
	}

	authPath := filepath.Join(codexHome, "auth.json")
	authStatus := "missing"
	if pathExists(authPath) {
		authStatus = "present"
	}

	if changed {
		fmt.Printf("Updated %s/config.toml to use file-based auth\n", codexHome)
	} else {
		fmt.Printf("Config already uses file-based auth (%s/config.toml)\n", codexHome)
	}
	fmt.Printf("Auth cache: %s\n", authStatus)
	if authStatus == "missing" {
		fmt.Println("Run `codex login` on the host to create ~/.codex/auth.json.")
	}
}

func loadConfig(path string) (Config, bool, error) {
	var cfg Config
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, false, nil
		}
		return cfg, false, fmt.Errorf("unable to read config file: %w", err)
	}
	if fi.IsDir() {
		return cfg, false, fmt.Errorf("config path is a directory: %s", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, false, fmt.Errorf("unable to read config file: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, false, fmt.Errorf("invalid config JSON: %w", err)
	}
	return cfg, true, nil
}

func loadRegistry(path string) (Registry, bool, error) {
	var reg Registry
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return reg, false, nil
		}
		return reg, false, fmt.Errorf("unable to read registry: %w", err)
	}
	if err := json.Unmarshal(data, &reg); err != nil {
		return reg, true, fmt.Errorf("invalid registry JSON: %w", err)
	}
	if reg.Items == nil {
		reg.Items = make(map[string]SplitRef)
	}
	return reg, true, nil
}

func saveRegistry(path string, reg Registry) error {
	if reg.Version == 0 {
		reg.Version = 1
	}
	if reg.Items == nil {
		reg.Items = make(map[string]SplitRef)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("unable to create registry directory: %w", err)
	}
	data, err := json.MarshalIndent(reg, "", "  ")
	if err != nil {
		return fmt.Errorf("unable to serialize registry: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), registryFileName+".tmp-*")
	if err != nil {
		return fmt.Errorf("unable to write registry: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("unable to write registry: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("unable to write registry: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("unable to save registry: %w", err)
	}
	return nil
}

func validateSourceRepo(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("source repo not found: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("source repo is not a directory: %s", path)
	}
	if !looksLikeGitRepo(path) {
		return fmt.Errorf("source repo does not look like a git repo: %s", path)
	}
	return nil
}

func validateDest(path string) error {
	if fi, err := os.Stat(path); err == nil {
		if !fi.IsDir() {
			return fmt.Errorf("destination exists and is not a directory: %s", path)
		}
		entries, err := os.ReadDir(path)
		if err != nil {
			return fmt.Errorf("unable to read destination directory: %w", err)
		}
		if len(entries) > 0 {
			return fmt.Errorf("destination directory is not empty: %s", path)
		}
		return nil
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("unable to check destination: %w", err)
	}

	parent := filepath.Dir(path)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return fmt.Errorf("unable to create destination parent: %w", err)
	}
	return nil
}

func looksLikeGitRepo(path string) bool {
	gitPath := filepath.Join(path, ".git")
	if info, err := os.Stat(gitPath); err == nil {
		return info.IsDir() || info.Mode().IsRegular()
	}
	headPath := filepath.Join(path, "HEAD")
	if info, err := os.Stat(headPath); err == nil && info.Mode().IsRegular() {
		if objInfo, err := os.Stat(filepath.Join(path, "objects")); err == nil && objInfo.IsDir() {
			return true
		}
		if _, err := os.Stat(filepath.Join(path, "packed-refs")); err == nil {
			return true
		}
	}
	return false
}

func defaultDestPath(src, base string) string {
	name := repoBaseName(src)
	stamp := time.Now().Format("20060102-150405")
	return filepath.Join(base, fmt.Sprintf("%s-%s", name, stamp))
}

func repoBaseName(path string) string {
	clean := filepath.Clean(path)
	base := filepath.Base(clean)
	if base == "." || base == string(filepath.Separator) || base == "" {
		return "repo"
	}
	return base
}

func registryPath(destBase string) string {
	return filepath.Join(destBase, registryFileName)
}

func ensureDocker() error {
	if _, err := exec.LookPath("docker"); err != nil {
		return errors.New("docker not found on PATH")
	}
	return nil
}

func ensureDockerImage(image, dockerfilePath, dockerfileContent string, buildIfMissing, forceBuild, dryRun bool) error {
	exists, err := dockerImageExists(image)
	if err != nil {
		return err
	}
	if exists && !forceBuild {
		return nil
	}
	if !exists && !buildIfMissing {
		return fmt.Errorf("docker image not found: %s", image)
	}

	var dockerfile string
	var context string
	var cleanup func()

	if strings.TrimSpace(dockerfilePath) != "" {
		dockerfile = resolvePath(dockerfilePath)
		if !pathExists(dockerfile) {
			return fmt.Errorf("dockerfile not found: %s", dockerfile)
		}
		context = filepath.Dir(dockerfile)
	} else {
		if strings.TrimSpace(dockerfileContent) == "" {
			return errors.New("embedded dockerfile is empty")
		}
		tempDir, err := os.MkdirTemp("", "repo-split-dockerfile-*")
		if err != nil {
			return fmt.Errorf("unable to create temp dir: %w", err)
		}
		cleanup = func() { _ = os.RemoveAll(tempDir) }
		dockerfile = filepath.Join(tempDir, "Dockerfile")
		if err := os.WriteFile(dockerfile, []byte(dockerfileContent), 0o644); err != nil {
			cleanup()
			return fmt.Errorf("unable to write dockerfile: %w", err)
		}
		context = tempDir
	}
	if cleanup != nil {
		defer cleanup()
	}

	args := []string{"build", "-f", dockerfile, "-t", image, context}
	if dryRun {
		fmt.Printf("docker %s\n", strings.Join(args, " "))
		return nil
	}
	cmd := exec.Command("docker", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker build failed: %w", err)
	}
	return nil
}

func dockerImageExists(image string) (bool, error) {
	out, err := exec.Command("docker", "image", "inspect", image).CombinedOutput()
	if err == nil {
		return true, nil
	}
	msg := strings.ToLower(string(out))
	if strings.Contains(msg, "no such image") || strings.Contains(msg, "not found") {
		return false, nil
	}
	return false, fmt.Errorf("docker image inspect failed: %s", strings.TrimSpace(string(out)))
}

func writeEnvFile(env []string, overrides map[string]string, denylist map[string]struct{}) (string, func(), error) {
	envMap := make(map[string]string, len(env))
	for _, kv := range env {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := parts[1]
		if key == "" || strings.ContainsAny(key, " \t\n\r") {
			continue
		}
		if _, denied := denylist[key]; denied {
			continue
		}
		if strings.ContainsAny(value, "\x00\n\r") {
			continue
		}
		envMap[key] = value
	}
	for key, value := range overrides {
		envMap[key] = value
	}
	if len(envMap) == 0 {
		return "", func() {}, nil
	}
	keys := make([]string, 0, len(envMap))
	for key := range envMap {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	file, err := os.CreateTemp("", "repo-split-env-*")
	if err != nil {
		return "", func() {}, fmt.Errorf("unable to create env file: %w", err)
	}
	for _, key := range keys {
		line := key + "=" + envMap[key]
		if _, err := file.WriteString(line + "\n"); err != nil {
			_ = file.Close()
			_ = os.Remove(file.Name())
			return "", func() {}, fmt.Errorf("unable to write env file: %w", err)
		}
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(file.Name())
		return "", func() {}, fmt.Errorf("unable to write env file: %w", err)
	}
	cleanup := func() {
		_ = os.Remove(file.Name())
	}
	return file.Name(), cleanup, nil
}

func containerEnvOverrides() map[string]string {
	term := strings.TrimSpace(os.Getenv("TERM"))
	if term == "" {
		term = "xterm-256color"
	}
	colorTerm := strings.TrimSpace(os.Getenv("COLORTERM"))
	if colorTerm == "" {
		colorTerm = "truecolor"
	}
	return map[string]string{
		"HOME":       "/root",
		"USER":       "root",
		"LOGNAME":    "root",
		"SHELL":      "/bin/bash",
		"PATH":       "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		"CODEX_HOME": "/root/.codex",
		"TERM":       term,
		"COLORTERM":  colorTerm,
	}
}

func containerEnvDenylist() map[string]struct{} {
	keys := []string{
		"HOME",
		"USER",
		"LOGNAME",
		"SHELL",
		"PATH",
		"PWD",
		"OLDPWD",
		"TMPDIR",
		"TMP",
		"TEMP",
		"SSH_AUTH_SOCK",
		"XDG_RUNTIME_DIR",
		"XDG_CONFIG_HOME",
		"XDG_DATA_HOME",
		"XDG_CACHE_HOME",
		"TERM_PROGRAM",
		"TERM_PROGRAM_VERSION",
		"TERM_SESSION_ID",
		"COLORTERM",
		"DISPLAY",
		"WAYLAND_DISPLAY",
		"XAUTHORITY",
	}
	deny := make(map[string]struct{}, len(keys))
	for _, key := range keys {
		deny[key] = struct{}{}
	}
	return deny
}

func ensureCodexAuthFileStore(codexHome string) (bool, error) {
	if codexHome == "" {
		return false, errors.New("codex home is empty")
	}
	if err := os.MkdirAll(codexHome, 0o755); err != nil {
		return false, fmt.Errorf("unable to create codex home: %w", err)
	}
	configPath := filepath.Join(codexHome, "config.toml")
	data, err := os.ReadFile(configPath)
	if err != nil && !os.IsNotExist(err) {
		return false, fmt.Errorf("unable to read codex config: %w", err)
	}

	contents := string(data)
	lines := strings.Split(contents, "\n")
	found := false
	changed := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "cli_auth_credentials_store") {
			found = true
			if !strings.Contains(trimmed, "\"file\"") {
				lines[i] = "cli_auth_credentials_store = \"file\""
				changed = true
			}
		}
	}
	if !found {
		lines = append(lines, "cli_auth_credentials_store = \"file\"")
		changed = true
	}

	if !changed {
		return false, nil
	}

	out := strings.Join(lines, "\n")
	if !strings.HasSuffix(out, "\n") {
		out += "\n"
	}
	if err := os.WriteFile(configPath, []byte(out), 0o644); err != nil {
		return false, fmt.Errorf("unable to write codex config: %w", err)
	}
	return true, nil
}

func containerState(name string) (bool, bool, error) {
	out, err := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", name).CombinedOutput()
	if err != nil {
		msg := strings.ToLower(string(out))
		if strings.Contains(msg, "no such") || strings.Contains(msg, "not found") {
			return false, false, nil
		}
		return false, false, fmt.Errorf("docker inspect failed: %s", strings.TrimSpace(string(out)))
	}
	running := strings.TrimSpace(string(out)) == "true"
	return true, running, nil
}

func ensureContainerRunning(name, image, repoPath, workspace string, hostNetwork, mountCodex, mountCodexHome bool, codexPath, envFile, awsPath, pgpassPath string, dryRun bool) error {
	exists, running, err := containerState(name)
	if err != nil {
		return err
	}
	if exists {
		if running {
			return nil
		}
		if dryRun {
			fmt.Printf("docker start %s\n", name)
			return nil
		}
		cmd := exec.Command("docker", "start", name)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("docker start failed: %w", err)
		}
		return nil
	}

	args := []string{"run", "-d", "--name", name}
	if hostNetwork {
		args = append(args, "--network", "host")
	}
	if envFile != "" {
		args = append(args, "--env-file", envFile)
	}
	args = append(args, "-v", fmt.Sprintf("%s:%s", repoPath, workspace), "-w", workspace)
	if awsPath != "" {
		args = append(args, "-v", fmt.Sprintf("%s:/root/.aws:ro", awsPath))
	}
	if pgpassPath != "" {
		args = append(args, "-v", fmt.Sprintf("%s:/root/.pgpass:ro", pgpassPath))
	}
	if mountCodex && codexPath != "" {
		args = append(args, "-v", fmt.Sprintf("%s:/usr/local/bin/codex:ro", codexPath))
	}
	if mountCodexHome {
		if codexHome := defaultCodexHome(); codexHome != "" {
			args = append(args, "-v", fmt.Sprintf("%s:/root/.codex", codexHome))
		}
	}
	args = append(args, image, "tail", "-f", "/dev/null")

	if dryRun {
		fmt.Printf("docker %s\n", strings.Join(args, " "))
		return nil
	}

	cmd := exec.Command("docker", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker run failed: %w", err)
	}
	return nil
}

func defaultCodexHome() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	path := filepath.Join(home, ".codex")
	if pathExists(path) {
		return path
	}
	return ""
}

func ensureNodeInContainer(name string, install bool) error {
	if containerHasCommand(name, "node") {
		return nil
	}
	if !install {
		return errors.New("node not found in container; re-run with -install-node or use an image with node")
	}
	commands := []string{
		"apt-get update -y",
		"apt-get install -y curl ca-certificates git",
		"curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
		"apt-get install -y nodejs",
	}
	for _, cmd := range commands {
		if err := dockerExecShell(name, cmd); err != nil {
			return err
		}
	}
	if !containerHasCommand(name, "node") {
		return errors.New("node install failed in container")
	}
	return nil
}

func ensureCodexInContainer(name string) error {
	if containerHasCommand(name, "codex") {
		if containerHasMount(name, "/usr/local/bin/codex") {
			return errors.New("codex is mounted from host; run repo-split container -id <id> -stop and retry without -mount-codex")
		}
		return nil
	}
	if err := dockerExecShell(name, "npm install -g @openai/codex"); err != nil {
		return err
	}
	if !containerHasCommand(name, "codex") {
		return errors.New("codex install failed in container")
	}
	return nil
}

func containerHasCommand(name, command string) bool {
	cmd := exec.Command("docker", "exec", name, "bash", "-lc", "command -v "+shellEscape(command))
	return cmd.Run() == nil
}

func containerHasMount(name, destination string) bool {
	cmd := exec.Command("docker", "inspect", "-f", "{{range .Mounts}}{{println .Destination}}{{end}}", name)
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if strings.TrimSpace(line) == destination {
			return true
		}
	}
	return false
}

func dockerExecShell(name, command string) error {
	cmd := exec.Command("docker", "exec", name, "bash", "-lc", command)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker exec failed: %w", err)
	}
	return nil
}

func shellEscape(value string) string {
	replacer := strings.NewReplacer(
		"\\", "\\\\",
		"\"", "\\\"",
		"$", "\\$",
		"`", "\\`",
	)
	return "\"" + replacer.Replace(value) + "\""
}

func generateID(reg Registry) string {
	for {
		id := randomID(idLength)
		if id == "" {
			continue
		}
		if !idExists(reg, id) {
			return id
		}
	}
}

func idExists(reg Registry, id string) bool {
	for _, entry := range reg.Items {
		if entry.ID == id {
			return true
		}
	}
	return false
}

func findByID(reg Registry, id string) (SplitRef, bool) {
	for _, entry := range reg.Items {
		if entry.ID == id {
			return entry, true
		}
	}
	return SplitRef{}, false
}

func randomID(length int) string {
	if length < 1 {
		return ""
	}
	const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
	buf := make([]byte, length)
	_, err := rand.Read(buf)
	if err != nil {
		return ""
	}
	for i, b := range buf {
		buf[i] = alphabet[int(b)%len(alphabet)]
	}
	return string(buf)
}

func ensureRegistryIDs(reg *Registry) bool {
	if reg.Items == nil {
		return false
	}
	changed := false
	for name, entry := range reg.Items {
		if entry.ID != "" && len(entry.ID) <= idLength {
			continue
		}
		entry.ID = generateID(*reg)
		reg.Items[name] = entry
		changed = true
	}
	return changed
}

func gitLastUpdated(path string) (time.Time, error) {
	cmd := exec.Command("git", "-C", path, "log", "-1", "--format=%ct")
	out, err := cmd.Output()
	if err != nil {
		return time.Time{}, err
	}
	value := strings.TrimSpace(string(out))
	if value == "" {
		return time.Time{}, errors.New("empty git log")
	}
	stamp, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return time.Time{}, err
	}
	return time.Unix(stamp, 0), nil
}

func ensureRemoteURL(destPath, srcPath, override string) error {
	remote := strings.TrimSpace(override)
	if remote == "" {
		if srcRemote, err := gitRemoteURL(srcPath, "origin"); err == nil {
			remote = srcRemote
		}
	}
	if remote == "" {
		return nil
	}
	if err := setGitRemote(destPath, "origin", remote); err != nil {
		return err
	}
	return nil
}

func gitRemoteURL(repoPath, remote string) (string, error) {
	cmd := exec.Command("git", "-C", repoPath, "remote", "get-url", remote)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func setGitRemote(repoPath, remote, url string) error {
	cmd := exec.Command("git", "-C", repoPath, "remote", "set-url", remote, url)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return nil
	}
	msg := strings.TrimSpace(string(out))
	if msg != "" {
		msg = ": " + msg
	}
	cmd = exec.Command("git", "-C", repoPath, "remote", "add", remote, url)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("unable to set remote %s%s", remote, msg)
	}
	return nil
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func humanDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	seconds := int64(d.Seconds())
	if seconds < 1 {
		return "0s"
	}
	days := seconds / 86400
	hours := (seconds % 86400) / 3600
	minutes := (seconds % 3600) / 60
	secs := seconds % 60

	switch {
	case days > 0:
		return fmt.Sprintf("%dd%dh", days, hours)
	case hours > 0:
		return fmt.Sprintf("%dh%dm", hours, minutes)
	case minutes > 0:
		return fmt.Sprintf("%dm%ds", minutes, secs)
	default:
		return fmt.Sprintf("%ds", secs)
	}
}

func expandPath(path string) string {
	if path == "" {
		return path
	}
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			if path == "~" {
				return home
			}
			if strings.HasPrefix(path, "~/") {
				return filepath.Join(home, strings.TrimPrefix(path, "~/"))
			}
		}
	}
	return os.ExpandEnv(path)
}

func resolvePath(path string) string {
	path = expandPath(path)
	if path == "" {
		return path
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	return abs
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func valueOrDefault[T any](v *T, def T) T {
	if v == nil {
		return def
	}
	return *v
}

func exitErr(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

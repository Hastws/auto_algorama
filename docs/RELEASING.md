## Release Checklist

Before creating a new release:

1. [ ] Update version number in `CMakeLists.txt`
2. [ ] Update `CHANGELOG.md` with all changes
3. [ ] Run all tests locally
4. [ ] Create and push a new tag

## Creating a Release

To create a new release, push a tag with the format `v*`:

```bash
# Update version
git add -A
git commit -m "Bump version to X.Y.Z"

# Create tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# Push tag to trigger release workflow
git push origin main --tags
```

The GitHub Actions workflow will automatically:
1. Build for Linux, macOS, and Windows
2. Run tests
3. Create a GitHub Release with all artifacts

## Artifacts

Each release includes:
- `AutoAlgorama-linux-x64.tar.gz` - Linux x64 binary
- `AutoAlgorama-macos-x64.zip` - macOS x64 binary
- `AutoAlgorama-windows-x64.zip` - Windows x64 binary with required DLLs

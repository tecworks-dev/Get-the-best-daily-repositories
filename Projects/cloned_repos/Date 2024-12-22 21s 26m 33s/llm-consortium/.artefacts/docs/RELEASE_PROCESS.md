# Release Process

This document outlines the process for creating and publishing new releases of the LLM Karpathy Consortium.

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## Release Checklist

### 1. Preparation
- [ ] Ensure all tests pass on main branch
- [ ] Check code coverage meets minimum threshold (>90%)
- [ ] Review open issues and PRs for critical fixes
- [ ] Update version number in:
  - `__init__.py`
  - `setup.py`
  - `docs/conf.py`

### 2. Documentation
- [ ] Update CHANGELOG.md
- [ ] Update API documentation if needed
- [ ] Review README.md for accuracy
- [ ] Update migration guide for breaking changes
- [ ] Check all documentation links

### 3. Testing
- [ ] Run full test suite:
```bash
pytest
pytest -m integration
pytest -m benchmark
```
- [ ] Test installation from clean environment:
```bash
pip install -e .
```
- [ ] Verify examples work

### 4. Create Release
```bash
# Create and push tag
git checkout main
git pull
git tag -a v0.x.x -m "Release v0.x.x"
git push origin v0.x.x

# Build distribution
python -m build
```

### 5. PyPI Publication
```bash
# Test PyPI first
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Verify test installation
pip install --index-url https://test.pypi.org/simple/ llm-karpathy-consortium

# Production PyPI
python -m twine upload dist/*
```

### 6. Post-Release
- [ ] Create GitHub release with changelog
- [ ] Announce release in discussions
- [ ] Update documentation version
- [ ] Verify PyPI page rendering
- [ ] Check GitHub Actions workflow success

## Hotfix Process

For critical bugs:

1. Create hotfix branch from latest release tag
```bash
git checkout -b hotfix/description v0.x.x
```

2. Apply fix and update version
3. Follow abbreviated release process
4. Merge changes back to main

## Release Schedule

- Major versions: As needed for breaking changes
- Minor versions: Monthly for new features
- Patch versions: As needed for bug fixes

## Version Support

- Latest version: Full support
- Previous minor version: Security updates
- Older versions: No support

## Emergency Releases

For critical security issues:

1. Assess severity and impact
2. Create emergency patch
3. Follow expedited release process
4. Notify users via security advisory

## Documentation

Each release should include:

1. What's New section
2. Breaking Changes (if any)
3. Migration Guide (if needed)
4. Known Issues
5. Future Plans

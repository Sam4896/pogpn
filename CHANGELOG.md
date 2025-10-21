# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
_Changes for the next version go here_

### Added
- 

### Changed
-


## [0.0.3] - 2025-10-21
### Added
- Coordinate-descent (CD) training for pathwise model via `POGPNPathwise.fit_torch_with_cd(...)`.
  - Cycles through non-root nodes in a deterministic topological order, updating one node per gradient descent step.
  - Compatible with existing joint training entry points (`fit(optimizer="torch"|"scipy")`).

## [0.0.2] - 2025-02-20
### Fixed
- Corrected an import bug in the `utils` module.


## [0.0.1] - 2025-01-15
### Added
- First release of the `pogpn` package.
- Basic DAG structure and model definitions.

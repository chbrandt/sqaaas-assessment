[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# SQAaaS Assessment CLI

A command-line tool for running software quality assessments using the [SQAaaS platform](https://sqaaas.eosc-synergy.eu), which evaluates source code repositories against quality criteria defined in the [EOSC-Synergy Software Quality Baseline](https://indigo-dc.github.io/sqa-baseline/).
This tool is composing the set of service Adapters created in the [EOSC Beyond](https://www.eosc-beyond.eu) project.

## Features

- **Automated Quality Assessment**: Run comprehensive quality checks on any Git repository
- **CLI and GitHub Actions**: Use as a standalone tool or integrate into CI/CD workflows
- **Configurable**: Customize API endpoints, timeouts, retry behavior, and custom quality steps
- **Rich Output**: Generate JSON reports, markdown summaries, and quality badges
- **Robust**: Built-in retry logic and error handling for reliable assessments

## Installation

### Requirements

- Python 3.7 or higher
- pip

### Dependencies

```bash
pip install requests jinja2
```

Or install from the repository:

```bash
git clone https://github.com/chbrandt/sqaaas-assessment.git
cd sqaaas-assessment
pip install -r requirements.txt  # if requirements.txt exists
```

## Quick Start

Run a quality assessment on a GitHub repository:

```bash
python assess.py https://github.com/user/repository
```

Save the report to a file:

```bash
python assess.py https://github.com/user/repository --output report.json
```

## Usage

### Basic Syntax

```bash
python assess.py <repository-url> [options]
```

### Command-Line Options

#### Required Arguments

- `repo` - Repository URL to assess (e.g., `https://github.com/user/repo`)

#### Optional Arguments

| Option            | Short | Description                                 | Default                                         |
| ----------------- | ----- | ------------------------------------------- | ----------------------------------------------- |
| `--branch`        | `-b`  | Branch name to assess                       | main/default branch                             |
| `--endpoint`      | `-e`  | SQAaaS API endpoint URL                     | `https://api-staging.sqaaas.eosc-synergy.eu/v1` |
| `--log-file`      |       | Write logs to specified file                | stderr only                                     |
| `--log-level`     | `-l`  | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO                                            |
| `--max-retries`   | `-m`  | Maximum retry attempts for API calls        | 3                                               |
| `--output`        | `-o`  | Save report JSON to file                    | stdout                                          |
| `--poll-interval` | `-p`  | Status poll interval in seconds             | 5                                               |
| `--quiet`         | `-q`  | Suppress non-error output                   | false                                           |
| `--retry-backoff` |       | Retry backoff multiplier                    | 2.0                                             |
| `--steps-file`    |       | JSON file with custom QC.Uni steps          | none                                            |
| `--summary-file`  | `-s`  | Write markdown summary to file              | none                                            |
| `--timeout`       | `-t`  | Request timeout in seconds                  | 60                                              |
| `--version`       | `-v`  | Show version information                    |                                                 |

### Examples

#### Basic Assessment

Assess the main branch of a repository:

```bash
python assess.py https://github.com/user/repo
```

#### Specific Branch

Assess a specific branch (e.g., "that_branch"):

```bash
python assess.py https://github.com/user/repo --branch that_branch
```

#### Save Outputs

Save both JSON report and markdown summary:

```bash
python assess.py https://github.com/user/repo \
  --output report.json \
  --summary-file summary.md
```

#### Debug Mode

Run with detailed logging:

```bash
python assess.py https://github.com/user/repo \
  --log-level DEBUG \
  --log-file assessment.log
```

#### Custom Configuration

Use custom API endpoint and timeouts:

```bash
python assess.py https://github.com/user/repo \
  --endpoint https://api.sqaaas.eosc-synergy.eu/v1 \
  --timeout 120 \
  --poll-interval 10 \
  --max-retries 5
```

#### With Custom Steps

Run assessment with custom unit testing steps:

```bash
python assess.py https://github.com/user/repo \
  --steps-file custom_steps.json
```

Example `custom_steps.json`:

```json
{
  "name": "tox_unit_step",
  "tool": "tox",
  "args": {
    "tox_env": "run_unit"
  }
}
```

## Configuration

### Environment Variables

You can configure the tool using environment variables (CLI arguments take precedence):

| Variable               | Description                        | Default                                         |
| ---------------------- | ---------------------------------- | ----------------------------------------------- |
| `SQAAAS_ENDPOINT`      | API endpoint URL                   | `https://api-staging.sqaaas.eosc-synergy.eu/v1` |
| `SQAAAS_TIMEOUT`       | Request timeout in seconds         | 60                                              |
| `SQAAAS_POLL_INTERVAL` | Status polling interval in seconds | 5                                               |
| `SQAAAS_MAX_RETRIES`   | Maximum retry attempts             | 3                                               |
| `SQAAAS_RETRY_BACKOFF` | Retry backoff multiplier           | 2.0                                             |

Example:

```bash
export SQAAAS_ENDPOINT=https://api.sqaaas.eosc-synergy.eu/v1
export SQAAAS_TIMEOUT=120
python assess.py https://github.com/user/repo
```

## Output Formats

### JSON Report

The complete assessment report in JSON format contains:

- Quality criteria results (pass/fail for each criterion)
- Evidence and assertions for each check
- Badge information (gold/silver/bronze)
- Repository metadata
- Links to quality standards

Example structure:

```json
{
  "report": {
    "QC.Acc": {
      "subcriteria": { ... }
    },
    "QC.Doc": {
      "subcriteria": { ... }
    }
  },
  "badge": {
    "software": {
      "criteria": {
        "gold": { "missing": [...] },
        "silver": { "missing": [...] },
        "bronze": { "missing": [...] }
      }
    }
  },
  "repository": [ ... ],
  "meta": { ... }
}
```

### Markdown Summary

The markdown summary includes:

- Quality criteria results table
- Achieved badge level
- Missing criteria for next badge level
- Links to detailed criteria documentation
- Link to full report in SQAaaS platform

### Quality Badges

Two types of badges are generated:

1. **SQAaaS Badge** - Official badge for achieved levels (gold/silver/bronze)
2. **shields.io Badge** - Dynamic status badge that updates with each assessment

Example badge markdown:

```markdown
[![SQAaaS badge shields.io](https://github.com/EOSC-synergy/<repo>.assess.sqaaas/raw/<branch>/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/<repo>.assess.sqaaas/<branch>/.report/assessment_output.json)
```

## Exit Codes

| Code | Description                        |
| ---- | ---------------------------------- |
| 0    | Success                            |
| 1    | Repository not defined or invalid  |
| 2    | Step workflow definition not found |
| 101  | HTTP/API error                     |
| 102  | General error                      |

## Quality Criteria

The SQAaaS platform evaluates repositories against these quality criteria:

- **QC.Acc** - Code Accessibility
- **QC.Wor** - Code Workflow
- **QC.Man** - Code Management
- **QC.Rev** - Code Review
- **QC.Ver** - Semantic Versioning
- **QC.Lic** - Licensing
- **QC.Met** - Code Metadata
- **QC.Doc** - Documentation
- **QC.Sty** - Code Style
- **QC.Uni** - Unit Testing
- **QC.Har** - Test Harness
- **QC.Tdd** - Test-Driven Development
- **QC.Sec** - Security
- **QC.Del** - Automated Delivery
- **QC.Dep** - Automated Deployment

See the [EOSC-Synergy Software Quality Baseline](https://indigo-dc.github.io/sqa-baseline/) for detailed criteria definitions.

## GitHub Actions Usage

This tool can also be used as a GitHub Action. For minimal integration:

```yaml
- uses: eosc-synergy/sqaaas-assessment-action@v2
```

For advanced GitHub Actions usage, see the [GitHub Actions documentation](https://github.com/EOSC-synergy/sqaaas-assessment-action/blob/main/docs/github-actions.md) (if available) or the original README.

## Development

### Code Style

This project uses [black](https://github.com/psf/black) for code formatting:

```bash
black assess.py
```

### Contributing

Contributions are welcome! Please ensure:

- Code follows the black style guide
- New features include documentation
- Tests pass (if applicable)
- Commit messages are clear and descriptive

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE.txt](LICENSE.txt) for details.

## Authors

- Pablo Orviz - Original author (GitHub Actions version)
- Carlos Brandt - CLI enhancements and standalone functionality

## Acknowledgments

This tool is part of the [EOSC Beyond](https://www.eosc-beyond.eu/) project and
uses the [SQAaaS platform](https://sqaaas.eosc-synergy.eu) for quality assessments.

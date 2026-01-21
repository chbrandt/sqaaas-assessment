# SPDX-FileCopyrightText: Copyright contributors to the Software Quality Assurance as a Service (SQAaaS) project.
# SPDX-FileContributor: Pablo Orviz <orviz@ifca.unican.es>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import requests

# Default logger (will be reconfigured based on CLI args)
logger = logging.getLogger("sqaaas-assessment-action")


# Custom Exceptions
class SQAaaSError(Exception):
    """Base exception for SQAaaS operations."""
    pass


class RepositoryError(SQAaaSError):
    """Repository validation or access errors."""
    pass


class PipelineError(SQAaaSError):
    """Pipeline execution errors."""
    pass


class APIError(SQAaaSError):
    """API communication errors."""
    pass


# Enums
class ExitCode(Enum):
    """Application exit codes."""
    REPO_NOT_DEFINED = 1
    STEP_WORKFLOW_NOT_FOUND = 2
    HTTP_ERROR = 101
    GENERAL_ERROR = 102


class PipelineStatus(Enum):
    """Pipeline execution status values."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNSTABLE = "UNSTABLE"
    ABORTED = "ABORTED"

    @classmethod
    def is_completed(cls, status: str) -> bool:
        """Check if status indicates pipeline completion.

        Args:
            status: Pipeline status string

        Returns:
            True if status indicates completion, False otherwise
        """
        return status in [
            cls.SUCCESS.value,
            cls.FAILURE.value,
            cls.UNSTABLE.value,
            cls.ABORTED.value
        ]

    @classmethod
    def is_successful(cls, status: str) -> bool:
        """Check if status indicates successful completion.

        Args:
            status: Pipeline status string

        Returns:
            True if status indicates success, False otherwise
        """
        return status in [cls.SUCCESS.value, cls.UNSTABLE.value]


# Dataclasses
@dataclass
class BadgeInfo:
    """Badge information for assessment results."""
    badge_sqaaas_md: Optional[str]
    badge_shields_md: str
    to_fulfill: List[str]
    next_level_badge: str


@dataclass
class ReportResult:
    """Individual quality report result."""
    status: bool
    assertion: str
    subcriterion: str
    criterion: str


@dataclass
class SQAaaSConfig:
    """Configuration for SQAaaS assessment."""
    endpoint: str = "https://api-staging.sqaaas.eosc-synergy.eu/v1"
    timeout: int = 30
    poll_interval: int = 5
    max_retries: int = 3
    retry_backoff: float = 2.0

    @classmethod
    def from_environment(cls) -> 'SQAaaSConfig':
        """Load configuration from environment variables.

        Returns:
            SQAaaSConfig instance with values from environment or defaults
        """
        return cls(
            endpoint=os.environ.get("SQAAAS_ENDPOINT", cls.endpoint),
            timeout=int(os.environ.get("SQAAAS_TIMEOUT", cls.timeout)),
            poll_interval=int(os.environ.get(
                "SQAAAS_POLL_INTERVAL", cls.poll_interval)),
            max_retries=int(os.environ.get("SQAAAS_MAX_RETRIES", cls.max_retries)),
            retry_backoff=float(os.environ.get(
                "SQAAAS_RETRY_BACKOFF", cls.retry_backoff)),
        )


# Legacy constants (maintained for compatibility during transition)
REPO_NOT_DEFINED_ERROR = ExitCode.REPO_NOT_DEFINED.value
STEP_WORKFLOW_NOT_FOUND_ERROR = ExitCode.STEP_WORKFLOW_NOT_FOUND.value
HTTP_ERROR_CODE = ExitCode.HTTP_ERROR.value
GENERAL_ERROR_CODE = ExitCode.GENERAL_ERROR.value

COMPLETED_STATUS = ["SUCCESS", "FAILURE", "UNSTABLE", "ABORTED"]
SUCCESFUL_STATUS = ["SUCCESS", "UNSTABLE"]
PIPELINE_STATUS_CHECK_INTERVAL = 5  # seconds

# API endpoint
LINKS_TO_STANDARD = {
    "QC.Acc": "https://indigo-dc.github.io/sqa-baseline/#code-accessibility-qc.acc",
    "QC.Wor": "https://indigo-dc.github.io/sqa-baseline/#code-workflow-qc.wor",
    "QC.Man": "https://indigo-dc.github.io/sqa-baseline/#code-management-qc.man",
    "QC.Rev": "https://indigo-dc.github.io/sqa-baseline/#code-review-qc.rev",
    "QC.Ver": "https://indigo-dc.github.io/sqa-baseline/#semantic-versioning-qc.ver",
    "QC.Lic": "https://indigo-dc.github.io/sqa-baseline/#licensing-qc.lic",
    "QC.Met": "https://indigo-dc.github.io/sqa-baseline/#code-metadata-qc.met",
    "QC.Doc": "https://indigo-dc.github.io/sqa-baseline/#documentation-qc.doc",
    "QC.Sty": "https://indigo-dc.github.io/sqa-baseline/#code-style-qc.sty",
    "QC.Uni": "https://indigo-dc.github.io/sqa-baseline/#unit-testing-qc.uni",
    "QC.Har": "https://indigo-dc.github.io/sqa-baseline/#test-harness-qc.har",
    "QC.Tdd": "https://indigo-dc.github.io/sqa-baseline/#test-driven-development-qc.tdd",
    "QC.Sec": "https://indigo-dc.github.io/sqa-baseline/#security-qc.sec",
    "QC.Del": "https://indigo-dc.github.io/sqa-baseline/#automated-delivery-qc.del",
    "QC.Dep": "https://indigo-dc.github.io/sqa-baseline/#automated-deployment-qc.dep",
}
# FIXME: add as CLI argument
ENDPOINT = "https://api-staging.sqaaas.eosc-synergy.eu/v1"

# Standard links
SYNERGY_BADGE_MARKDOWN = {
    "gold": {
        "sqaaas": '[![SQAaaS badge](https://github.com/EOSC-synergy/SQAaaS/raw/master/badges/badges_150x116/badge_software_gold.png)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/{repo}.assess.sqaaas/{branch}/.report/assessment_output.json "SQAaaS gold badge achieved")',
    },
    "silver": {
        "sqaaas": '[![SQAaaS badge](https://github.com/EOSC-synergy/SQAaaS/raw/master/badges/badges_150x116/badge_software_silver.png)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/{repo}.assess.sqaaas/{branch}/.report/assessment_output.json "SQAaaS silver badge achieved")',
    },
    "bronze": {
        "sqaaas": '[![SQAaaS badge](https://github.com/EOSC-synergy/SQAaaS/raw/master/badges/badges_150x116/badge_software_bronze.png)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/{repo}.assess.sqaaas/{branch}/.report/assessment_output.json "SQAaaS bronze badge achieved")',
    },
}
SHIELDS_BADGE_MARKDOWN = "[![SQAaaS badge shields.io](https://github.com/EOSC-synergy/{repo}.assess.sqaaas/raw/{branch}/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/{repo}.assess.sqaaas/{branch}/.report/assessment_output.json)"

SUMMARY_TEMPLATE = """## SQAaaS results :bellhop_bell:

### Quality criteria summary
| Result | Assertion | Subcriterion ID | Criterion ID |
| ------ | --------- | --------------- | ------------ |
{%- for result in report_results %}
| {{ ":heavy_check_mark:" if result.status else ":heavy_multiplication_x:" }} | {{ result.assertion }} | {{ result.subcriterion }} | {{ result.criterion }} |
{%- endfor %}

### Quality badge
{%- if badge_results.badge_sqaaas_md %}
 - SQAaaS-based badge: {{ badge_results.badge_sqaaas_md }}
{%- endif %}
shields.io-based badge: {{ badge_results.badge_shields_md }}
{%- if badge_results.next_level_badge %}
 - Missing quality criteria for next level badge ({{ badge_results.next_level_badge }}): {% for criterion_to_fulfill in badge_results.to_fulfill %}[`{{ criterion_to_fulfill }}`]({{ links_to_standard[criterion_to_fulfill] }}) {% endfor %}
{%- endif %}

### :clipboard: __View full report in the [SQAaaS platform]({{ report_url }})__
"""


# API Client
class SQAaaSAPIClient:
    """Client for interacting with the SQAaaS API."""

    def __init__(self, config: SQAaaSConfig):
        """Initialize API client with configuration.

        Args:
            config: SQAaaS configuration object
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request_with_retry(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            payload: Optional JSON payload for POST requests

        Returns:
            Response object from the API

        Raises:
            APIError: If request fails after all retries
        """
        method = method.upper()
        url = f"{self.config.endpoint}/{path}"

        for attempt in range(self.config.max_retries):
            try:
                if method == "POST":
                    response = self.session.post(
                        url, json=payload, timeout=self.config.timeout
                    )
                else:
                    response = self.session.get(url, timeout=self.config.timeout)

                response.raise_for_status()
                logger.debug(f"Request to {path} succeeded")
                return response

            except requests.HTTPError as http_err:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"HTTP error on {path} after {self.config.max_retries} attempts: {http_err}")
                    raise APIError(f"HTTP error: {http_err}") from http_err
                wait_time = self.config.retry_backoff ** attempt
                logger.warning(
                    f"HTTP error on {path} (attempt {attempt + 1}/{self.config.max_retries}): {http_err}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except requests.RequestException as req_err:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Request error on {path} after {self.config.max_retries} attempts: {req_err}")
                    raise APIError(f"Request error: {req_err}") from req_err
                wait_time = self.config.retry_backoff ** attempt
                logger.warning(
                    f"Request error on {path} (attempt {attempt + 1}/{self.config.max_retries}): {req_err}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise APIError(
            f"Request to {path} failed after {self.config.max_retries} attempts")

    def create_pipeline(self, payload: Dict[str, Any]) -> str:
        """Create assessment pipeline.

        Args:
            payload: Pipeline configuration payload

        Returns:
            Pipeline ID string

        Raises:
            APIError: If pipeline creation fails
            PipelineError: If response doesn't contain expected data
        """
        logger.info("Creating assessment pipeline")
        response = self._request_with_retry("POST", "pipeline/assessment", payload)
        response_data = response.json()

        if "id" not in response_data:
            raise PipelineError("Pipeline creation response missing 'id' field")

        pipeline_id = response_data["id"]
        logger.info(f"Created pipeline with ID: {pipeline_id}")
        return pipeline_id

    def run_pipeline(self, pipeline_id: str) -> None:
        """Trigger pipeline execution.

        Args:
            pipeline_id: Pipeline ID to run

        Raises:
            APIError: If pipeline execution trigger fails
        """
        logger.info(f"Triggering execution of pipeline {pipeline_id}")
        self._request_with_retry("POST", f"pipeline/{pipeline_id}/run")

    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status.

        Args:
            pipeline_id: Pipeline ID to check

        Returns:
            Dictionary containing status information

        Raises:
            APIError: If status retrieval fails
            PipelineError: If response doesn't contain expected data
        """
        response = self._request_with_retry("GET", f"pipeline/{pipeline_id}/status")
        response_data = response.json()

        if "build_status" not in response_data:
            raise PipelineError("Status response missing 'build_status' field")

        return response_data

    def get_pipeline_output(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline assessment results.

        Args:
            pipeline_id: Pipeline ID to get results for

        Returns:
            Dictionary containing assessment report JSON

        Raises:
            APIError: If retrieving output fails
        """
        logger.info(f"Retrieving output for pipeline {pipeline_id}")
        response = self._request_with_retry(
            "GET", f"pipeline/assessment/{pipeline_id}/output")
        return response.json()


def validate_repo_url(repo: str) -> bool:
    """Validate repository URL format.

    Args:
        repo: Repository URL to validate

    Returns:
        True if URL is valid, False otherwise
    """
    if not repo or not repo.strip():
        return False
    # Basic validation: should start with http:// or https://
    return repo.startswith(("http://", "https://"))


def create_payload(
    repo: str,
    branch: Optional[str] = None,
    step_tools: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> str:
    """Create JSON payload for triggering SQAaaS assessment.

    Args:
        repo: Repository URL to assess
        branch: Branch name to assess (optional)
        step_tools: Dictionary mapping criterion IDs to their tool configurations

    Returns:
        JSON string representation of the payload

    Raises:
        ValueError: If repository URL is invalid
    """
    if not validate_repo_url(repo):
        raise ValueError(f"Invalid repository URL: {repo}")

    if step_tools is None:
        step_tools = {}

    payload: Dict[str, Any] = {
        "repo_code": {
            "repo": repo,
            "branch": branch,
        }
    }
    if step_tools:
        for criterion, tools in step_tools.items():
            payload["criteria_workflow"] = [{"id": criterion, "tools": tools}]
            break  # FIXME: only interested in the first one, i.e. QC.Uni
    logger.debug(f"Payload for triggering SQAaaS assessment: {payload}")

    return json.dumps(payload)


def sqaaas_request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None
) -> requests.Response:
    """Make an HTTP request to the SQAaaS API endpoint.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: API endpoint path
        payload: Optional JSON payload for POST requests

    Returns:
        Response object from the API

    Raises:
        APIError: If request fails due to HTTP or connection errors
    """
    if payload is None:
        payload = {}

    method = method.upper()
    headers = {"Content-Type": "application/json"}
    args: Dict[str, Any] = {
        "method": method,
        "url": f"{ENDPOINT}/{path}",
        "headers": headers
    }
    if method in ["POST"]:
        args["json"] = payload

    try:
        response = requests.request(**args)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
        logger.info("Success!")
        return response
    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise APIError(f"HTTP error: {http_err}") from http_err
    except requests.RequestException as req_err:
        logger.error(f"Request error occurred: {req_err}")
        raise APIError(f"Request error: {req_err}") from req_err
    except Exception as err:
        logger.error(f"Unexpected error occurred: {err}")
        raise APIError(f"Unexpected error: {err}") from err


def _create_pipeline(
    repo: str,
    branch: Optional[str] = None,
    step_tools: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> str:
    """Create assessment pipeline and return pipeline ID.

    Args:
        repo: Repository URL to assess
        branch: Branch name to assess (optional)
        step_tools: Dictionary mapping criterion IDs to their tool configurations

    Returns:
        Pipeline ID string

    Raises:
        APIError: If pipeline creation fails
        PipelineError: If response doesn't contain expected data
    """
    payload = json.loads(create_payload(repo, branch, step_tools))
    logger.debug(f"Using payload: {payload}")

    response = sqaaas_request("post", "pipeline/assessment", payload=payload)
    response_data = response.json()

    if "id" not in response_data:
        raise PipelineError("Pipeline creation response missing 'id' field")

    pipeline_id = response_data["id"]
    logger.info(f"Created pipeline with ID: {pipeline_id}")
    return pipeline_id


def _run_pipeline(pipeline_id: str) -> None:
    """Trigger pipeline execution.

    Args:
        pipeline_id: Pipeline ID to run

    Raises:
        APIError: If pipeline execution trigger fails
    """
    logger.info(f"Running pipeline {pipeline_id}")
    sqaaas_request("post", f"pipeline/{pipeline_id}/run")


def _poll_pipeline_status(pipeline_id: str) -> str:
    """Poll pipeline until completion and return final status.

    Args:
        pipeline_id: Pipeline ID to poll

    Returns:
        Final pipeline status string

    Raises:
        APIError: If status polling fails
        PipelineError: If response doesn't contain expected data
    """
    while True:
        response = sqaaas_request("get", f"pipeline/{pipeline_id}/status")
        response_data = response.json()

        if "build_status" not in response_data:
            raise PipelineError("Status response missing 'build_status' field")

        build_status = response_data["build_status"]

        if PipelineStatus.is_completed(build_status):
            logger.info(f"Pipeline {pipeline_id} finished with status {build_status}")
            return build_status

        logger.info(
            f"Current status is {build_status}. Waiting {PIPELINE_STATUS_CHECK_INTERVAL} seconds.."
        )
        time.sleep(PIPELINE_STATUS_CHECK_INTERVAL)


def _get_pipeline_output(pipeline_id: str) -> Dict[str, Any]:
    """Retrieve pipeline assessment results.

    Args:
        pipeline_id: Pipeline ID to get results for

    Returns:
        Dictionary containing assessment report JSON

    Raises:
        APIError: If retrieving output fails
    """
    logger.info(f"Retrieving output for pipeline {pipeline_id}")
    response = sqaaas_request("get", f"pipeline/assessment/{pipeline_id}/output")
    return response.json()


def run_assessment(
    repo: str,
    branch: Optional[str] = None,
    step_tools: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """Run SQAaaS assessment pipeline for a repository.

    This function orchestrates the full lifecycle of a pipeline assessment:
    creating the pipeline, running it, polling for status, and retrieving results.

    Args:
        repo: Repository URL to assess
        branch: Branch name to assess (optional)
        step_tools: Dictionary mapping criterion IDs to their tool configurations

    Returns:
        Dictionary containing the complete assessment report JSON

    Raises:
        APIError: If any API operation fails
        PipelineError: If pipeline operations fail
    """
    if step_tools is None:
        step_tools = {}

    # Create pipeline
    pipeline_id = _create_pipeline(repo, branch, step_tools)

    # Run pipeline
    _run_pipeline(pipeline_id)

    # Poll for completion
    final_status = _poll_pipeline_status(pipeline_id)

    # Get results
    sqaaas_report_json = _get_pipeline_output(pipeline_id)

    return sqaaas_report_json


# Report Processor
class ReportProcessor:
    """Process SQAaaS assessment reports and generate summaries."""

    def extract_results(self, report_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract quality results from report.

        Args:
            report_json: Complete assessment report JSON

        Returns:
            List of result dictionaries with status, assertion, subcriterion, criterion
        """
        report_results: List[Dict[str, Any]] = []
        for criterion, criterion_data in report_json["report"].items():
            for subcriterion, subcriterion_data in criterion_data["subcriteria"].items():
                for evidence in subcriterion_data["evidence"]:
                    # Sanitize message for markdown
                    msg = evidence["message"]
                    msg = msg.replace("<", "_")
                    msg = msg.replace(">", "_")
                    msg = msg.replace("\n", "<br />")

                    report_results.append({
                        "status": evidence["valid"],
                        "assertion": msg,
                        "subcriterion": subcriterion,
                        "criterion": criterion,
                    })
        return report_results

    def extract_badge_info(self, report_json: Dict[str, Any]) -> BadgeInfo:
        """Extract badge information from report.

        Args:
            report_json: Complete assessment report JSON

        Returns:
            BadgeInfo dataclass with badge details
        """
        badge_software = report_json["badge"]["software"]
        repo_data = report_json["repository"][0]  # NOTE: temporarily use first element
        repo = os.path.basename(repo_data["name"])
        branch = repo_data["tag"]

        badge_shields_md = SHIELDS_BADGE_MARKDOWN.format(repo=repo, branch=branch)
        badge_sqaaas_md: Optional[str] = None
        to_fulfill: List[str] = []
        next_level_badge = ""

        for badgeclass in ["gold", "silver", "bronze"]:
            missing = badge_software["criteria"][badgeclass]["missing"]
            if not missing:
                logger.debug(f"Not missing criteria: achieved {badgeclass} badge")
                badge_share_data = SYNERGY_BADGE_MARKDOWN[badgeclass]
                badge_sqaaas_md = badge_share_data["sqaaas"].format(
                    repo=repo, branch=branch
                )
                break
            else:
                to_fulfill = missing
                next_level_badge = badgeclass
                logger.debug(
                    f"Missing criteria found ({to_fulfill}) for {badgeclass} badge, "
                    f"going one level down"
                )

        return BadgeInfo(
            badge_sqaaas_md=badge_sqaaas_md,
            badge_shields_md=badge_shields_md,
            to_fulfill=to_fulfill,
            next_level_badge=next_level_badge
        )

    def generate_summary(self, report_json: Dict[str, Any]) -> str:
        """Generate markdown summary from report.

        Args:
            report_json: Complete assessment report JSON

        Returns:
            Markdown-formatted summary string
        """
        report_results = self.extract_results(report_json)
        badge_info = self.extract_badge_info(report_json)

        full_report_url = "/".join([
            "https://sqaaas.eosc-synergy.eu/#/full-assessment/report",
            report_json["meta"]["report_json_url"],
        ])

        template = jinja2.Environment().from_string(SUMMARY_TEMPLATE)
        return template.render(
            report_results=report_results,
            badge_results={
                "badge_sqaaas_md": badge_info.badge_sqaaas_md,
                "badge_shields_md": badge_info.badge_shields_md,
                "to_fulfill": badge_info.to_fulfill,
                "next_level_badge": badge_info.next_level_badge,
            },
            report_url=full_report_url,
            links_to_standard=LINKS_TO_STANDARD,
        )


def get_summary(sqaaas_report_json: Dict[str, Any]) -> str:
    """Generate a formatted summary from SQAaaS assessment report.

    Args:
        sqaaas_report_json: Complete assessment report JSON from SQAaaS

    Returns:
        Markdown-formatted summary string for GitHub Step Summary
    """
    processor = ReportProcessor()
    return processor.generate_summary(sqaaas_report_json)


def write_summary(sqaaas_report_json: Dict[str, Any]) -> str:
    """Write assessment summary to GitHub Step Summary.

    Args:
        sqaaas_report_json: Complete assessment report JSON from SQAaaS

    Returns:
        Formatted summary string that was written
    """
    summary = get_summary(sqaaas_report_json)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        logger.info("Setting GITHUB_STEP_SUMMARY environment variable")
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            print(summary, file=f)
            logger.info("Summary data added to GITHUB_STEP_SUMMARY")
    else:
        logger.warning("Cannot set GITHUB_STEP_SUMMARY")

    return summary


def get_repo_data() -> Tuple[str, str]:
    """Extract repository and branch information from environment variables.

    Checks for custom INPUT_REPO/INPUT_BRANCH first, then falls back to
    GitHub Actions environment variables if assessing the current repository.

    Returns:
        Tuple of (repository_url, branch_name)
    """
    repo = os.environ.get("INPUT_REPO", "")
    branch = os.environ.get("INPUT_BRANCH", "")
    if repo:
        logger.info(f"Not assessing current repository: {repo}")
    else:
        logger.debug(f"Assessing current repository: {repo}")
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if repo:
            repo = os.path.join("https://github.com", repo)
        if not branch:
            event_name = os.environ.get("GITHUB_EVENT_NAME")
            if event_name in ["pull_request"]:
                logger.debug(
                    "Getting branch name from GITHUB_HEAD_REF as it was triggered by a pull request"
                )
                branch = os.environ.get("GITHUB_HEAD_REF", "")
            else:
                logger.debug("Getting branch name from GITHUB_REF_NAME")
                branch = os.environ.get("GITHUB_REF_NAME", "")

    return (repo, branch)


def get_custom_steps() -> Dict[str, List[Dict[str, Any]]]:
    """Load custom step workflow definitions from JSON files.

    Reads step workflow definitions for QC.Uni criterion from JSON files
    specified in the INPUT_QC_UNI_STEPS environment variable.

    Returns:
        Dictionary mapping criterion IDs to their tool configurations

    Raises:
        SystemExit: If a specified workflow definition file is not found
    """
    custom_steps: Dict[str, List[Dict[str, Any]]] = {}
    # QC.Uni
    step_workflows = os.environ.get("INPUT_QC_UNI_STEPS", "")
    step_names = step_workflows.split()
    step_tools: List[Dict[str, Any]] = []
    if step_workflows:
        for step_name in step_names:
            _step_payload_file = f"{step_name}.json"
            if not os.path.exists(_step_payload_file):
                logger.error(
                    f"Aborting..step workflow definition not found: {step_name}"
                )
                sys.exit(STEP_WORKFLOW_NOT_FOUND_ERROR)
            logger.debug(f"Step workflow found: {step_name}")
            with open(_step_payload_file, "r") as f:
                step_tools.append(json.load(f))

        custom_steps = {"QC.Uni": step_tools}

    return custom_steps


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),
        ] + ([logging.FileHandler(log_file)] if log_file else [])
    )

    # Update module logger
    global logger
    logger = logging.getLogger("sqaaas-assessment-action")
    logger.setLevel(level)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog='sqaaas-assess',
        description='Run SQAaaS quality assessment on a repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --repo https://github.com/user/repo
  %(prog)s -r https://github.com/user/repo -b develop -o report.json
  %(prog)s -r https://github.com/user/repo --log-level DEBUG
  %(prog)s -r https://github.com/user/repo --summary-file summary.md

Exit Codes:
  0   - Success
  1   - Repository not defined
  2   - Step workflow not found
  101 - HTTP/API error
  102 - General error

Environment Variables:
  SQAAAS_ENDPOINT      - API endpoint (overridden by --endpoint)
  SQAAAS_TIMEOUT       - Request timeout (overridden by --timeout)
  SQAAAS_POLL_INTERVAL - Status poll interval (overridden by --poll-interval)
  SQAAAS_MAX_RETRIES   - Maximum retry attempts (overridden by --max-retries)
  SQAAAS_RETRY_BACKOFF - Retry backoff multiplier (overridden by --retry-backoff)
        '''
    )

    # Required arguments
    parser.add_argument(
        '--repo', '-r',
        type=str,
        required=True,
        help='Repository URL to assess (e.g., https://github.com/user/repo)'
    )

    # Optional arguments
    parser.add_argument(
        '--branch', '-b',
        type=str,
        help='Branch name to assess (default: main or from git)'
    )
    parser.add_argument(
        '--endpoint', '-e',
        type=str,
        default='https://api-staging.sqaaas.eosc-synergy.eu/v1',
        help='SQAaaS API endpoint (default: %(default)s)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for report JSON'
    )
    parser.add_argument(
        '--summary-file', '-s',
        type=str,
        help='Write markdown summary to file'
    )
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: %(default)s)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Write logs to file'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=30,
        help='Request timeout in seconds (default: %(default)s)'
    )
    parser.add_argument(
        '--poll-interval', '-p',
        type=int,
        default=5,
        help='Status poll interval in seconds (default: %(default)s)'
    )
    parser.add_argument(
        '--max-retries', '-m',
        type=int,
        default=3,
        help='Maximum retry attempts (default: %(default)s)'
    )
    parser.add_argument(
        '--retry-backoff',
        type=float,
        default=2.0,
        help='Retry backoff multiplier (default: %(default)s)'
    )
    parser.add_argument(
        '--steps-file',
        type=str,
        help='JSON file with custom QC.Uni steps'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser.parse_args()


def load_steps_from_file(steps_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load custom steps from JSON file.

    Args:
        steps_file: Path to JSON file containing steps

    Returns:
        Dictionary mapping criterion IDs to tool configurations

    Raises:
        FileNotFoundError: If steps file doesn't exist
        json.JSONDecodeError: If steps file is invalid JSON
    """
    if not os.path.exists(steps_file):
        raise FileNotFoundError(f"Steps file not found: {steps_file}")

    with open(steps_file, 'r') as f:
        steps_data = json.load(f)

    return {"QC.Uni": [steps_data]}


def main_cli() -> None:
    """CLI entry point for standalone usage."""
    args = parse_arguments()

    # Configure logging
    setup_logging(args.log_level, args.log_file)

    if args.quiet:
        logger.setLevel(logging.ERROR)

    try:
        # Create configuration from CLI arguments
        config = SQAaaSConfig(
            endpoint=args.endpoint,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff
        )

        logger.info(f"Starting SQAaaS assessment for repository: {args.repo}")

        # Load custom steps if provided
        step_tools: Dict[str, List[Dict[str, Any]]] = {}
        if args.steps_file:
            logger.info(f"Loading custom steps from: {args.steps_file}")
            step_tools = load_steps_from_file(args.steps_file)

        # Run assessment
        sqaaas_report_json = run_assessment(
            repo=args.repo,
            branch=args.branch,
            step_tools=step_tools
        )

        if not sqaaas_report_json:
            logger.error("Could not get report data from SQAaaS platform")
            sys.exit(ExitCode.GENERAL_ERROR.value)

        logger.info("SQAaaS assessment completed successfully")

        # Save report JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(sqaaas_report_json, f, indent=2)
            logger.info(f"Report JSON saved to: {args.output}")

        # Generate and save summary if requested
        if args.summary_file:
            processor = ReportProcessor()
            summary = processor.generate_summary(sqaaas_report_json)
            with open(args.summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"Summary markdown saved to: {args.summary_file}")

        # Print summary to stdout if not quiet
        if not args.quiet and not args.output:
            processor = ReportProcessor()
            summary = processor.generate_summary(sqaaas_report_json)
            print(summary)

    except (APIError, PipelineError, RepositoryError) as e:
        logger.error(f"Assessment error: {e}")
        sys.exit(ExitCode.HTTP_ERROR.value if isinstance(
            e, APIError) else ExitCode.GENERAL_ERROR.value)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(ExitCode.GENERAL_ERROR.value)


def main_github_action() -> None:
    """GitHub Actions entry point (environment variable mode).

    Orchestrates the entire assessment process: retrieving repository data,
    loading custom steps, running the assessment, and writing the summary.
    Handles all exceptions and exits with appropriate error codes.
    """
    # Configure logging for GitHub Actions
    logging.basicConfig(level=logging.DEBUG)

    try:
        repo, branch = get_repo_data()
        if not repo:
            logger.error(
                "Repository URL for the assessment not defined through "
                "INPUT_REPO: cannot continue"
            )
            sys.exit(ExitCode.REPO_NOT_DEFINED.value)

        logger.info(f"Trigger SQAaaS assessment with code repository: {repo}")

        # Get any JSON step payload being generated by sqaaas-step-action
        step_tools = get_custom_steps()

        # Run assessment
        sqaaas_report_json = run_assessment(
            repo=repo, branch=branch, step_tools=step_tools)

        if sqaaas_report_json:
            logger.info("SQAaaS assessment data obtained. Creating summary..")
            logger.debug(sqaaas_report_json)
            summary = write_summary(sqaaas_report_json)
            if summary:
                logger.debug(summary)
        else:
            logger.warning("Could not get report data from SQAaaS platform")

    except APIError as e:
        logger.error(f"API error occurred: {e}")
        sys.exit(ExitCode.HTTP_ERROR.value)
    except (PipelineError, RepositoryError) as e:
        logger.error(f"Pipeline/Repository error occurred: {e}")
        sys.exit(ExitCode.GENERAL_ERROR.value)
    except SQAaaSError as e:
        logger.error(f"SQAaaS error occurred: {e}")
        sys.exit(ExitCode.GENERAL_ERROR.value)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        sys.exit(ExitCode.GENERAL_ERROR.value)


def main() -> None:
    """Main entry point - routes to CLI or GitHub Actions mode."""
    if len(sys.argv) > 1:
        # CLI mode (arguments provided)
        main_cli()
    else:
        # GitHub Actions mode (environment variables)
        main_github_action()


if __name__ == "__main__":
    main()

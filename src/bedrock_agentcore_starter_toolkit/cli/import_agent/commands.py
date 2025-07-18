# pylint: disable=line-too-long
"""Bedrock Agent Translation Tool."""

import os
import uuid
import json
import subprocess  # nosec # needed to run the agent file
import sys
import traceback

import boto3
import typer
import questionary
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...services.import_agent.scripts.bedrock_to_langchain import BedrockLangchainTranslation
from ...services.import_agent.scripts.bedrock_to_strands import BedrockStrandsTranslation
from ...services.import_agent.utils import auth_and_get_info, get_agent_aliases, get_agents, get_clients
from ..common import console

app = typer.Typer(help="Import Agent")


def _verify_aws_credentials() -> bool:
    """Verify that AWS credentials are present and valid."""
    try:
        # Try to get the caller identity to verify credentials
        boto3.client("sts").get_caller_identity()
        return True
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]AWS credentials are invalid![/bold red]\n"
                f"Error: {str(e)}\n"
                f"Please reconfigure your AWS credentials by running:\n"
                f"[bold]aws configure[/bold]",
                title="Authentication Error",
                border_style="red",
            )
        )
        return False


def _run_agent(output_path):
    """Run the generated agent."""
    try:
        console.print(
            Panel(
                "[bold green]Launching the agent...[/bold green]\nYou can start using your translated agent below:",
                title="Agent Launch",
                border_style="green",
            )
        )

        # Run the agent file with subprocess under the current CLI
        process = subprocess.Popen([sys.executable, output_path, "--cli"])  # nosec

        while True:
            try:
                process.wait()
                break
            except KeyboardInterrupt:
                pass

        console.print("\n[green]Agent execution completed.[/green]")

    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Failed to run the agent![/bold red]\nError: {str(e)}",
                title="Execution Error",
                border_style="red",
            )
        )


def _agentcore_invoke_cli():
    """Run the generated agent."""
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            console.print("\n[yellow]Exiting AgentCore CLI...[/yellow]")
            break

        try:
            os.system("agentcore invoke " + json.dumps({"message": query}))  # nosec
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Error invoking agent![/bold red]\nError: {str(e)}",
                    title="Invocation Error",
                    border_style="red",
                )
            )
            continue


@app.command()
def import_agent():
    """Migrate a Bedrock Agent to LangChain or Strands."""
    try:
        output_dir = "./output/"  # Default output directory for generated code

        os.makedirs(output_dir, exist_ok=True)

        # Display welcome banner
        console.print(
            Panel(
                Text("Bedrock Agent Translation Tool", style="bold cyan"),
                subtitle="Convert your Bedrock Agent to LangChain/Strands code with AgentCore Primitives",
                border_style="cyan",
            )
        )

        # Verify AWS credentials
        console.print("[bold]Verifying AWS credentials...[/bold]")
        if not _verify_aws_credentials():
            return

        console.print("[bold green]✓[/bold green] AWS credentials verified!")

        # Get AWS credentials and clients
        credentials = boto3.Session().get_credentials()
        bedrock_client, bedrock_agent_client = get_clients(credentials)

        # Get all agents in the user's account
        console.print("[bold]Fetching available agents...[/bold]")
        agents = get_agents(bedrock_agent_client)

        if not agents:
            console.print(
                Panel("[bold red]No agents found in your account![/bold red]", title="Error", border_style="red")
            )
            return

        # Display agents in a table
        agents_table = Table(title="\nAvailable Agents")
        agents_table.add_column("ID", style="cyan")
        agents_table.add_column("Name", style="green")
        agents_table.add_column("Description", style="yellow")

        for agent in agents:
            agents_table.add_row(agent["id"], agent["name"] or "No name", agent["description"] or "No description")

        console.print(agents_table, "\n")

        # Let user select an agent
        agent_choices = [f"{agent['name']} ({agent['id']})" for agent in agents]
        selected_agent = questionary.select(
            "Select an agent:",
            choices=agent_choices,
        ).ask()

        if selected_agent is None:  # Handle case where user presses Esc
            console.print("\n[yellow]Agent selection cancelled by user.[/yellow]")
            return

        # Extract agent ID from selection
        agent_id = selected_agent.split("(")[-1].strip(")")

        # Get all aliases for the selected agent
        console.print(f"[bold]Fetching aliases for agent {agent_id}...[/bold]")
        aliases = get_agent_aliases(bedrock_agent_client, agent_id)

        if not aliases:
            console.print(
                Panel(
                    f"[bold red]No aliases found for agent {agent_id}![/bold red]",
                    title="Error",
                    border_style="red",
                )
            )
            return

        # Display aliases in a table
        aliases_table = Table(title=f"\nAvailable Aliases for Agent {agent_id}")
        aliases_table.add_column("ID", style="cyan")
        aliases_table.add_column("Name", style="green")
        aliases_table.add_column("Description", style="yellow")

        for alias in aliases:
            aliases_table.add_row(alias["id"], alias["name"] or "No name", alias["description"] or "No description")

        console.print(aliases_table, "\n")

        # Let user select an alias
        alias_choices = [f"{alias['name']} ({alias['id']})" for alias in aliases]
        selected_alias = questionary.select(
            "Select an alias:",
            choices=alias_choices,
        ).ask()

        if selected_alias is None:  # Handle case where user presses Esc
            console.print("\n[yellow]Alias selection cancelled by user.[/yellow]")
            return

        # Extract alias ID from selection
        agent_alias_id = selected_alias.split("(")[-1].strip(")")

        # Select target platform
        target_platform = questionary.select(
            "Select your target platform:",
            choices=["langchain", "strands"],
        ).ask()

        if target_platform is None:  # Handle case where user presses Esc
            console.print("\n[yellow]Platform selection cancelled by user.[/yellow]")
            return

        # Show progress
        with console.status("[bold green]Fetching agent configuration...[/bold green]"):
            try:
                agent_config = auth_and_get_info(agent_id, agent_alias_id, output_dir)
                console.print("[bold green]✓[/bold green] Agent configuration retrieved!")
            except Exception as e:
                console.print(
                    Panel(
                        f"[bold red]Failed to retrieve agent configuration![/bold red]\nError: {str(e)} {traceback.print_exc()}",
                        title="Configuration Error",
                        border_style="red",
                    )
                )
                return

        debug = questionary.confirm("Would you like to enable verbose mode?", default=False).ask()

        if debug is None:  # Handle case where user presses Esc
            console.print("\n[yellow]Debug selection cancelled by user.[/yellow]")
            return

        # Ask about primitives to opt into
        primitive_options = [
            # {"name": "AgentCore Gateway (Tools for your agent)", "value": "gateway"},
            {"name": "AgentCore Memory (Maintain conversation context)", "value": "memory"},
            {"name": "AgentCore Code Interpreter (Run code in your agent)", "value": "code_interpreter"},
            {"name": "AgentCore Observability (Logging and monitoring)", "value": "observability"},
        ]

        # Default to all AgentCore primitives enabled
        primitives_opt_in = {
            # "gateway": False,
            "memory": False,
            "code_interpreter": False,
            "observability": False,
        }

        selected_primitives = questionary.checkbox(
            "Select AgentCore primitives to include:", choices=[option["name"] for option in primitive_options]
        ).ask()
        for option in primitive_options:
            if option["name"] in selected_primitives:
                primitives_opt_in[option["value"]] = True

        console.print(
            f"[bold green]✓[/bold green] Selected primitives: {[k for k, v in primitives_opt_in.items() if v]}"
        )

        if selected_primitives is None:  # Handle case where user presses Esc
            console.print("\n[yellow]Primitives selection cancelled by user.[/yellow]")
            return

        # Translate the agent
        with console.status(f"[bold green]Translating agent to {target_platform}...[/bold green]"):
            try:
                if target_platform == "langchain":
                    output_path = os.path.join(output_dir, "langchain_agent.py")
                    translator = BedrockLangchainTranslation(
                        agent_config, debug=debug, output_dir=output_dir, enabled_primitives=primitives_opt_in
                    )
                    translator.translate_bedrock_to_langchain(output_path)
                else:  # strands
                    output_path = os.path.join(output_dir, "strands_agent.py")
                    translator = BedrockStrandsTranslation(
                        agent_config, debug=debug, output_dir=output_dir, enabled_primitives=primitives_opt_in
                    )
                    translator.translate_bedrock_to_strands(output_path)

                console.print(f"[bold green]✓[/bold green] Agent translated to {target_platform}!")
                console.print(f"[bold]  Output file:[/bold] {output_path}")
            except Exception as e:
                console.print(
                    Panel(
                        f"[bold red]Failed to translate agent![/bold red]\nError: {str(e), traceback.print_exc()}",
                        title="Translation Error",
                        border_style="red",
                    )
                )
                return

        # AgentCore Runtime deployment options
        deploy_agentcore_runtime = questionary.confirm(
            "Would you like to deploy the agent to AgentCore Runtime? (This will take a few minutes)", default=False
        ).ask()

        output_path = os.path.abspath(output_path)
        output_dir = os.path.abspath(output_dir)

        if deploy_agentcore_runtime is None:  # Handle case where user presses Esc
            console.print("\n[yellow]AgentCore Runtime deployment selection cancelled by user.[/yellow]")

        if deploy_agentcore_runtime:
            try:
                agent_name = f"agent_{uuid.uuid4().hex[:8].lower()}"
                console.print("[bold]  \nDeploying agent to AgentCore Runtime...\n[/bold]")
                os.system(
                    f"cd {output_dir} && agentcore configure --entrypoint {output_path} --requirements-file requirements.txt --ecr auto -n '{agent_name}' && agentcore configure set-default '{agent_name}' && agentcore launch"  # nosec
                )  # nosec

            except Exception as e:
                console.print(
                    Panel(
                        f"[bold red]Failed to deploy agent to AgentCore Runtime![/bold red]\nError: {str(e)}",
                        title="Deployment Error",
                        border_style="red",
                    )
                )
                return

        run_options = ["Run locally", "Don't run now"]

        if deploy_agentcore_runtime:
            run_options.insert(1, "Run on AgentCore Runtime")

        run_agent_choice = questionary.select(
            "How would you like to run the agent?",
            choices=run_options,
        ).ask()
        if run_agent_choice is None:  # Handle case where user presses Esc
            console.print("\n[yellow]Run selection cancelled by user.[/yellow]")
            return

    except KeyboardInterrupt:
        console.print("\n[yellow]Migration process cancelled by user.[/yellow]")
    except SystemExit:
        console.print("\n[yellow]Migration process exited.[/yellow]")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]An unexpected error occurred![/bold red]\nError: {str(e)}",
                title="Unexpected Error",
                border_style="red",
            )
        )

    if run_agent_choice == "Run locally":
        _run_agent(output_path)
    elif run_agent_choice == "Run on AgentCore Runtime" and deploy_agentcore_runtime:
        console.print(
            Panel(
                "[bold green]Starting AgentCore Runtime interactive CLI...[/bold green]",
                title="AgentCore Runtime",
                border_style="green",
            )
        )
        _agentcore_invoke_cli()
    else:
        console.print(
            Panel(
                f"[bold green]Migration completed successfully![/bold green]\n"
                f"You can run your agent later with:\n"
                f"[bold]python {output_path}[/bold]",
                title="Migration Complete",
                border_style="green",
            )
        )


# ruff: noqa

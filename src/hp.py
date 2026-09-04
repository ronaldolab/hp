"""Homopolymer (HP) chromosome-simulation workflows.

This module consolidates the equilibration, expansion, and capsule workflows. It deliberately preserves their force parameters and block counts from the paper. Only the OpenMiChroM 1.1.1 keyword names have been updated.

The HP simulation workflows use mostly OpenMiChroM 1.1.1 and OpenMM 8.3.1.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# The package source lives in ``src/OpenMiChroM/OpenMiChroM`` so that its setup.py remains self-contained. Make that package root precede the outer directory when this module is run directly from ``src``.
_OPENMICROM_SOURCE_ROOT = Path(__file__).resolve().parent / "OpenMiChroM"
if (_OPENMICROM_SOURCE_ROOT / "OpenMiChroM" / "__init__.py").is_file():
    sys.path.insert(0, str(_OPENMICROM_SOURCE_ROOT))

from OpenMiChroM.ChromDynamics import MiChroM  # noqa: I001


Model = Literal["knotted", "unknotted"]


@dataclass
class HPConfig:
    """Configuration for an HP simulation workflow.

    Length and force parameters use the OpenMiChroM units. They are not converted in this class.
    """

    output_folder: Path
    chrom_sequence: Path
    model: Model = "knotted"
    platform: str = "CPU"
    blocks: int = 1000 #3000
    time_step: float = 0.01
    equilibration_time: int = 5000
    equilibrated_structure: Path | None = None
    types_table_dir: Path | None = None
    capsule_radius: float | None = None
    capsule_force_constant: float = 30.0
    sphere_radius: float = 20.0

    collapse_blocks: int = 200
    production_blocks: int = 200
    expansion_blocks: int = 5000
    expansion_start_block: int = 100
    untie_blocks: int = 12000
    capsule_relaxation_blocks: int = 2000
    capsule_production_blocks: int = 12000
    sphere_hot_blocks: int = 500
    sphere_relaxation_blocks: int = 5000
    sphere_schedule_steps: int = 500


@dataclass
class HP:
    """Run one HP simulation workflow.

    Available modes are ``collapse``, ``equilibration``, ``production``,
    ``all_at_once``, ``sphere_all_at_once``, ``expansion``, ``untie``,
    and ``capsule-collapse``. The two ``*_all_at_once`` workflows run collapse,
    equilibration, and production in sequence.
    """

    config: HPConfig
    radius_of_gyration: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # OpenMiChroM prefixes explicit structure filenames with its output folder. Keep this path absolute so a relative folder is not added twice when those filenames are generated.
        self.config.output_folder = Path(self.config.output_folder).expanduser().resolve()
        self.config.chrom_sequence = Path(self.config.chrom_sequence)
        if self.config.model not in {"knotted", "unknotted"}:
            raise ValueError("model must be 'knotted' or 'unknotted'")

    @property
    def _temperature(self) -> float:
        return {"knotted": 1.5, "unknotted": 0.7}[self.config.model]

    @property
    def _repulsive_cutoff(self) -> float:
        return 500.0 if self.config.model == "unknotted" else 3.0

    def _new_simulation(self, name: str, temperature: float) -> MiChroM:
        """Create a MiChroM simulation with the configured platform."""
        simulation = MiChroM(name=name, temperature=temperature,
                             timeStep=self.config.time_step)
        simulation.setup(platform=self.config.platform)
        simulation.saveFolder(str(self.config.output_folder))
        return simulation

    def _add_polymer_forces(
        self,
        simulation: MiChroM,
        *,
        repulsive_cutoff: float,
        flat_bottom: bool = False,
        type_to_type: bool = False,
        ideal_chromosome: bool = False,
        capsule: tuple[float, float, float] | None = None,
    ) -> None:
        """Add HP forces using OpenMiChroM 1.1.1 parameter names."""
        simulation.addFENEBonds(kFb=30.0)
        simulation.addAngles(kA=2.0)
        simulation.addRepulsiveSoftCore(eCut=repulsive_cutoff)

        if flat_bottom:
            simulation.addFlatBottomHarmonic(kR=5e-3, nRad=15.0)
        if type_to_type:
            simulation.addTypetoType(mu=3.22, rc=1.78)
        if capsule is not None:
            radius, half_length, force_constant = capsule
            simulation.addCapsuleConfinement(
                r_conf=radius, z_conf=half_length, kr=force_constant
            )
        if ideal_chromosome:
            simulation.addIdealChromosome()

    def _create_context(self, simulation: MiChroM) -> None:
        """Create the OpenMM context after all initial forces are defined."""
        simulation.createSimulation()

    def _configure_trajectory(
        self, simulation: MiChroM, save_every_blocks: int
    ) -> None:
        """Attach the OpenMiChroM 1.1.1 CNDB trajectory reporter."""
        simulation.createReporters(
            statistics=True,
            traj=True,
            trajFormat="cndb",
            outputName="traj",
            interval=self.config.blocks * save_every_blocks,
        )

    def _configure_statistics(
        self, simulation: MiChroM, save_every_blocks: int = 1
    ) -> None:
        """Attach a statistics reporter without writing a trajectory."""
        simulation.createReporters(
            statistics=True,
            traj=False,
            interval=self.config.blocks * save_every_blocks,
        )

    def _run_block(self, simulation: MiChroM) -> None:
        """Run one configured block; trajectory output is reporter-managed."""
        simulation.run(self.config.blocks, report=True)

    def _load_structure(self, simulation: MiChroM, structure_file: Path) -> None:
        if not structure_file.is_file():
            raise FileNotFoundError(f"Structure file does not exist: {structure_file}")
        suffix = structure_file.suffix.lower()
        if suffix == ".gro":
            chromosome = simulation.initStructure(
                CoordFiles=str(structure_file),
                ChromSeq=str(self.config.chrom_sequence),
                isRing=True,
                mode="gro",
            )
        elif suffix == ".ndb":
            chromosome = simulation.initStructure(
                CoordFiles=str(structure_file), isRing=True, mode="ndb"
            )
        else:
            raise ValueError(
                f"Unsupported structure format '{structure_file.suffix}'; use .gro or .ndb"
            )
        simulation.loadStructure(chromosome, center=True)

    @property
    def _collapsed_structure(self) -> Path:
        return self.config.output_folder / "1.collapse_final.gro"

    @property
    def _equilibration_output_structure(self) -> Path:
        return self.config.output_folder / "2.equilibration_final.gro"

    @property
    def _production_final_structure(self) -> Path:
        return self.config.output_folder / "3.production_final.gro"

    def _record_rg(self, simulation: MiChroM) -> None:
        positions = simulation.getPositions()
        centered_positions = positions - np.mean(positions, axis=0)
        self.radius_of_gyration.append(
            float(np.sqrt(np.sum(np.var(centered_positions, axis=0))))
        )

    def _close_storage(self, simulation: MiChroM) -> None:
        """Compatibility no-op for reporter-managed OpenMiChroM trajectories."""
        del simulation

    def _equilibrated_structure(self) -> Path:
        structure = self.config.equilibrated_structure
        if structure is None:
            raise ValueError("equilibrated_structure is required for this workflow")
        structure = Path(structure)
        if not structure.is_file():
            raise FileNotFoundError(f"Equilibrated structure does not exist: {structure}")
        return structure

    @staticmethod
    def _natural_sort_key(path: Path) -> list[object]:
        return [int(item) if item.isdigit() else item.lower()
                for item in re.split(r"(\d+)", path.name)]

    def _type_tables(self) -> list[Path]:
        directory = self.config.types_table_dir
        if directory is None:
            raise ValueError("types_table_dir is required for expansion")
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Type-table directory does not exist: {directory}")
        tables = sorted(
            directory.glob("Michrom_repulsive_*.ff"), key=self._natural_sort_key
        )
        if not tables:
            raise FileNotFoundError(f"No Michrom_repulsive_*.ff tables in {directory}")
        return tables

    def _change_custom_type(self, simulation: MiChroM, table: Path) -> None:
        simulation.removeForce("CustomTypes")
        simulation.addAdditionalForce(
            simulation.addCustomTypes, mu=3.22, rc=1.78, TypesTable=str(table)
        )
        print(f"Changing to type table: {table.name}")

    def collapse(self) -> MiChroM:
        """Run the 200-block spring-spiral collapse stage."""
        if not self.config.chrom_sequence.is_file():
            raise FileNotFoundError(f"Chromosome sequence does not exist: {self.config.chrom_sequence}")
        simulation = self._new_simulation("1.collapse", temperature=1.0)
        chromosome = simulation.createSpringSpiral(
            ChromSeq=str(self.config.chrom_sequence), isRing=True
        )
        simulation.loadStructure(chromosome, center=True)
        self._add_polymer_forces(
            simulation, repulsive_cutoff=self._repulsive_cutoff,
            flat_bottom=True, type_to_type=True,
        )
        self._create_context(simulation)
        self._configure_statistics(simulation)
        simulation.saveStructure(fileName="1.collapse_initial.gro", mode="gro")
        for _ in range(self.config.collapse_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
        simulation.saveStructure(fileName=str(self._collapsed_structure), mode="gro")
        self._finalize_stage_statistics(simulation)
        return simulation

    def equilibrate(self, structure_file: Path | None = None) -> MiChroM:
        """Equilibrate a collapsed structure for ``equilibration_time`` blocks."""
        source = structure_file or self._collapsed_structure
        simulation = self._new_simulation("2.equilibration", self._temperature)
        self._load_structure(simulation, Path(source))
        self._add_polymer_forces(
            simulation, repulsive_cutoff=self._repulsive_cutoff,
            flat_bottom=True, type_to_type=True,
        )
        self._create_context(simulation)
        self._configure_statistics(simulation)
        for _ in range(self.config.equilibration_time):
            self._run_block(simulation)
            self._record_rg(simulation)
        simulation.saveStructure(
            fileName=str(self._equilibration_output_structure), mode="gro"
        )
        self._finalize_stage_statistics(simulation)
        return simulation

    def production(self, structure_file: Path | None = None) -> MiChroM:
        """Run the unconstrained 200-block production simulation."""
        source = structure_file or self._equilibration_output_structure
        simulation = self._new_simulation("3.production", self._temperature)
        self._load_structure(simulation, Path(source))
        self._add_polymer_forces(
            simulation, repulsive_cutoff=self._repulsive_cutoff, type_to_type=True
        )
        self._create_context(simulation)
        self._configure_trajectory(simulation, save_every_blocks=1)
        for _ in range(self.config.production_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
        self._close_storage(simulation)
        simulation.saveStructure(
            fileName=str(self._production_final_structure), mode="gro"
        )
        self._finalize_stage_statistics(simulation)
        return simulation

    def all_at_once(self) -> MiChroM:
        """Run collapse, equilibration, then production."""
        self.collapse()
        self.equilibrate()
        return self.production()

    def expansion(self) -> MiChroM:
        """Run the type-table expansion workflow."""
        tables = self._type_tables()
        if self.config.expansion_blocks <= self.config.expansion_start_block:
            raise ValueError("expansion_blocks must exceed expansion_start_block")
        source = self._equilibrated_structure()
        simulation = self._new_simulation("3.expansion", self._temperature)
        self._load_structure(simulation, source)
        self._add_polymer_forces(
            simulation, repulsive_cutoff=100.0, type_to_type=True
        )
        self._create_context(simulation)
        simulation.saveStructure(fileName="3.expansion_initial.gro", mode="gro")
        self._configure_trajectory(simulation, save_every_blocks=10)

        total_expansion_blocks = (
            self.config.expansion_blocks - self.config.expansion_start_block
        )
        change_interval = total_expansion_blocks // len(tables)
        if change_interval < 1:
            raise ValueError("Expansion has fewer blocks than type tables")

        for current_block in range(self.config.expansion_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
            if current_block % 500 == 0:
                simulation.saveStructure(mode="gro")

            if current_block == self.config.expansion_start_block:
                print("Expanding structure")
                simulation.removeForce("TypetoType")
                simulation.addAdditionalForce(
                    simulation.addCustomTypes, mu=3.22, rc=1.78,
                    TypesTable=str(tables[0]),
                )
                print(f"Changing to type table: {tables[0].name}")
            elif current_block > self.config.expansion_start_block:
                elapsed = current_block - self.config.expansion_start_block
                if elapsed % change_interval == 0:
                    table_index = elapsed // change_interval
                    if table_index < len(tables):
                        self._change_custom_type(simulation, tables[table_index])

        self._close_storage(simulation)
        simulation.saveStructure(mode="gro")
        return simulation

    def untie(self) -> MiChroM:
        """Run the capsule-confined, ideal-chromosome untying workflow."""
        radius = self._require_capsule_radius()
        simulation = self._new_simulation("3.untie", temperature=1.0)
        self._load_structure(simulation, self._equilibrated_structure())
        self._add_polymer_forces(
            simulation, repulsive_cutoff=3.0, ideal_chromosome=True,
            capsule=(radius, radius, self.config.capsule_force_constant),
        )
        self._create_context(simulation)
        simulation.saveStructure(fileName="3.untie_initial.gro", mode="gro")
        self._configure_trajectory(simulation, save_every_blocks=10)
        for current_block in range(self.config.untie_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
            if current_block % 500 == 0:
                simulation.saveStructure(mode="gro")
        self._close_storage(simulation)
        simulation.saveStructure(mode="gro")
        return simulation

    def capsule_collapse(self) -> MiChroM:
        """Collapse into the shrinking capsule, then run production.

        The capsule radius follows the existing exponential schedule from 21.2 to 10.4, stopping the relaxation once the configured target radius is reached. No units or schedule values are changed here.
        """
        target_radius = self._require_capsule_radius()
        simulation = self._new_simulation("3.capsule_collapse", temperature=1.0)
        self._load_structure(simulation, self._equilibrated_structure())
        self._add_polymer_forces(
            simulation, repulsive_cutoff=self._repulsive_cutoff,
            capsule=(21.2, 21.2, self.config.capsule_force_constant),
        )
        self._create_context(simulation)
        simulation.saveStructure(fileName="3.capsule_collapse_initial.gro", mode="gro")

        radius_schedule = 10.4 + (21.2 - 10.4) * np.exp(-np.linspace(0, 4, 50))
        previous_radius: float | None = None
        for current_block in range(self.config.capsule_relaxation_blocks):
            schedule_index = min(
                int(current_block * len(radius_schedule) /
                    self.config.capsule_relaxation_blocks),
                len(radius_schedule) - 1,
            )
            current_radius = float(radius_schedule[schedule_index])
            if current_radius < target_radius:
                continue
            self._run_block(simulation)
            if previous_radius is None or not np.isclose(
                current_radius, previous_radius, rtol=1e-8
            ):
                simulation.context.setParameter("r_conf", current_radius)
                simulation.context.setParameter("z_conf", current_radius)
                previous_radius = current_radius

        self._configure_trajectory(simulation, save_every_blocks=50)
        for current_block in range(self.config.capsule_production_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
            if current_block % 500 == 0:
                simulation.saveStructure(mode="gro")
        self._close_storage(simulation)
        simulation.saveStructure(mode="gro")
        return simulation

    def sphere_all_at_once(self) -> MiChroM:
        """Run the spherical collapse, equilibration, and production workflow.

        The first stage starts from a spring spiral, expands for 500 blocks at
        ``T = 3.0``, and compresses from radius 600.0 using the existing
        exponential schedule. After removing spherical confinement, the model
        equilibrates with type-to-type interactions and then performs the
        standard production length. The sphere radius and all schedule
        parameters remain in the OpenMiChroM units used by this protocol.
        """
        target_radius = self._require_sphere_radius()
        if not self.config.chrom_sequence.is_file():
            raise FileNotFoundError(f"Chromosome sequence does not exist: {self.config.chrom_sequence}")

        simulation = self._new_simulation("1.collapse", temperature=1.0)
        chromosome = simulation.createSpringSpiral(
            ChromSeq=str(self.config.chrom_sequence), isRing=True
        )
        simulation.loadStructure(chromosome, center=True)
        self._add_polymer_forces(
            simulation,
            repulsive_cutoff=self._repulsive_cutoff,
            # The ideal-chromosome term is part of the unknotted model only.
            # The knotted branch retains the same spherical-collapse schedule
            # without this lengthwise interaction.
            ideal_chromosome=(self.config.model == "unknotted"),
        )
        self._create_context(simulation)

        hot_temperature = 3.0
        simulation.integrator.setTemperature(hot_temperature / 0.008314)
        for _ in range(self.config.sphere_hot_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
        simulation.saveStructure(
            fileName=str(self.config.output_folder / "1.collapse_initial.gro"),
            mode="gro",
        )

        simulation.integrator.setTemperature(1.0 / 0.008314)
        simulation.addAdditionalForce(
            simulation.addSphericalConfinement, r_conf=600.0, k_conf=30.0
        )

        final_schedule_radius = target_radius * 0.78
        radius_schedule = final_schedule_radius + (600.0 - final_schedule_radius) * np.exp(
            -np.linspace(0, 5, self.config.sphere_schedule_steps)
        )
        previous_radius: float | None = None
        for current_block in range(self.config.sphere_relaxation_blocks):
            schedule_index = min(
                int(current_block * len(radius_schedule) /
                    self.config.sphere_relaxation_blocks),
                len(radius_schedule) - 1,
            )
            current_radius = float(radius_schedule[schedule_index])
            if current_radius < target_radius:
                continue
            self._run_block(simulation)
            self._record_rg(simulation)
            if previous_radius is None or not np.isclose(
                current_radius, previous_radius, rtol=1e-8
            ):
                simulation.context.setParameter("r_sphere", current_radius)
                previous_radius = current_radius

        simulation.saveStructure(fileName=str(self._collapsed_structure), mode="gro")

        # Stage 2: retain the sphere-collapse forces (including the
        # unknotted-only ideal-chromosome term) and replace spherical
        # confinement with type-to-type interactions for equilibration.
        simulation.removeForce("SphericalConfinement")
        simulation.addAdditionalForce(simulation.addTypetoType)
        for _ in range(self.config.equilibration_time):
            self._run_block(simulation)
            self._record_rg(simulation)
        simulation.saveStructure(
            fileName=str(self._equilibration_output_structure), mode="gro"
        )

        # Stage 3: generate the production trajectory from the equilibrated
        # spherical-collapse state using the existing production block count.
        simulation.name = "3.production"
        self._configure_trajectory(simulation, save_every_blocks=1)
        for _ in range(self.config.production_blocks):
            self._run_block(simulation)
            self._record_rg(simulation)
        self._close_storage(simulation)
        simulation.saveStructure(
            fileName=str(self._production_final_structure), mode="gro"
        )
        self._finalize_stage_statistics(simulation)
        return simulation

    def _require_capsule_radius(self) -> float:
        radius = self.config.capsule_radius
        if radius is None or radius <= 0:
            raise ValueError("capsule_radius must be a positive value")
        return radius

    def _require_sphere_radius(self) -> float:
        radius = self.config.sphere_radius
        if radius is None or radius <= 0:
            raise ValueError("sphere_radius must be a positive value")
        return radius

    def save_radius_of_gyration(self) -> Path:
        """Write the radius-of-gyration values accumulated by this instance."""
        output_path = self.config.output_folder / "Rg.data"
        np.savetxt(output_path, self.radius_of_gyration)
        return output_path

    def _finalize_stage_statistics(self, simulation: MiChroM) -> list[Path]:
        """Rename the shared OpenMiChroM statistics files after one stage."""
        renamed_paths: list[Path] = []
        for filename in ("initialStats.txt", "statistics.txt"):
            source_path = self.config.output_folder / filename
            if not source_path.is_file():
                continue

            destination_path = (
                self.config.output_folder / f"{simulation.name}_{filename}"
            )
            source_path.rename(destination_path)
            renamed_paths.append(destination_path)
        return renamed_paths

    def run(self, mode: str) -> MiChroM:
        """Run one named workflow and write its accumulated Rg values."""
        workflows = {
            "collapse": self.collapse,
            "equilibration": self.equilibrate,
            "production": self.production,
            "all_at_once": self.all_at_once,
            "sphere_all_at_once": self.sphere_all_at_once,
            "expansion": self.expansion,
            "untie": self.untie,
            "capsule-collapse": self.capsule_collapse,
        }
        try:
            simulation = workflows[mode]()
        except KeyError as error:
            valid = ", ".join(workflows)
            raise ValueError(f"Unknown mode '{mode}'. Choose one of: {valid}") from error
        self.save_radius_of_gyration()
        return simulation


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("collapse", "equilibration", "production", "all_at_once", "expansion",
                 "sphere_all_at_once", "untie", "capsule-collapse"),
    )
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("chrom_sequence", type=Path)
    parser.add_argument("--model", choices=("knotted", "unknotted"), default="knotted")
    parser.add_argument("--platform", default="CPU")
    parser.add_argument("--equilibration-time", type=int, default=5000,
                        help="Number of equilibration blocks (default: 5000).")
    parser.add_argument("--equilibrated-structure", type=Path,
                        help="Single equilibrated GRO or NDB structure for expansion and capsule modes.")
    parser.add_argument("--types-table-dir", type=Path)
    parser.add_argument("--capsule-radius", type=float)
    parser.add_argument("--sphere-radius", type=float, default=20.0,
                        help="Target radius for sphere_all_at_once (default: 20.0).")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    configuration = HPConfig(
        output_folder=arguments.output_folder,
        chrom_sequence=arguments.chrom_sequence,
        model=arguments.model,
        platform=arguments.platform,
        equilibration_time=arguments.equilibration_time,
        equilibrated_structure=arguments.equilibrated_structure,
        types_table_dir=arguments.types_table_dir,
        capsule_radius=arguments.capsule_radius,
        sphere_radius=arguments.sphere_radius,
    )
    HP(configuration).run(arguments.mode)

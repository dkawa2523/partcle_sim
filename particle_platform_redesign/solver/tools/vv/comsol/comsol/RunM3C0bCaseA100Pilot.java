import com.comsol.model.DatasetFeature;
import com.comsol.model.ExportFeature;
import com.comsol.model.Model;
import com.comsol.model.NumericalFeature;
import com.comsol.model.SolverFeature;
import com.comsol.model.StudyFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Locale;

/**
 * Generates the Brownian-off, full-deterministic M3-C0b pre-event step pilot.
 *
 * <p>The audited theory MPH is loaded from an isolated copy for every step
 * size.  No model is saved.  All deterministic particle forces and dynamic
 * charging remain active; only Brownian and the already-inactive Saffman lift
 * are explicitly disabled. This pilot intentionally covers Case A, 100 nm,
 * through the last common saved time before the first boundary event observed
 * in the preceding 10/5/2.5 us characterization. Matrix expansion is performed
 * only after this finer three-step run is accepted.
 */
public final class RunM3C0bCaseA100Pilot {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = "fptas";
  private static final String BACKGROUND_STUDY = "stdASf";
  private static final String BACKGROUND_SOLUTION = "sol26";
  private static final String TLIST = "range(0[s],1e-5[s],4.5e-4[s])";
  private static final int[] STEP_CODES = {625, 3125, 15625};

  private static final String[] STATE_COLUMNS = {
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s",
    "charge_rate_e_per_s",
    "particle_mass_kg"
  };

  private static final String[] STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1", "1", "s", "1/s", "kg"
  };

  private static final String[] FORCE_COLUMNS = {
    "particle_id",
    "time_s",
    "electric_force_r_N",
    "electric_force_z_N",
    "ion_drag_force_r_N",
    "ion_drag_force_z_N",
    "epstein_force_r_N",
    "epstein_force_z_N",
    "thermophoretic_force_r_N",
    "thermophoretic_force_z_N",
    "lift_force_r_N",
    "lift_force_z_N",
    "dep_force_r_N",
    "dep_force_z_N",
    "gravity_buoyancy_force_r_N",
    "gravity_buoyancy_force_z_N"
  };

  private static final String[] FORCE_UNITS = {
    "1", "s", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N"
  };

  private static final String[] NEUTRAL_COLUMNS = {
    "particle_id",
    "time_s",
    "gas_velocity_r_m_per_s",
    "gas_velocity_z_m_per_s",
    "gas_temperature_K",
    "gas_density_kg_per_m3",
    "gas_dynamic_viscosity_Pa_s",
    "gas_thermal_conductivity_W_per_mK",
    "gas_mean_free_path_m",
    "temperature_gradient_r_K_per_m",
    "temperature_gradient_z_K_per_m",
    "effective_heat_flux_r_W_per_m2",
    "effective_heat_flux_z_W_per_m2",
    "azimuthal_vorticity_per_s",
    "particle_diameter_m"
  };

  private static final String[] NEUTRAL_UNITS = {
    "1", "s", "m/s", "m/s", "K", "kg/m^3", "Pa*s", "W/(m*K)", "m", "K/m", "K/m",
    "W/m^2", "W/m^2", "1/s", "m"
  };

  private static final String[] ELECTRIC_COLUMNS = {
    "particle_id",
    "time_s",
    "electric_field_r_V_per_m",
    "electric_field_z_V_per_m",
    "electric_field_squared_V2_per_m2",
    "gradient_E2_r_V2_per_m3",
    "gradient_E2_z_V2_per_m3",
    "particle_surface_potential_per_charge_V"
  };

  private static final String[] ELECTRIC_UNITS = {
    "1", "s", "V/m", "V/m", "V^2/m^2", "V^2/m^3", "V^2/m^3", "V"
  };

  private static final String[] PLASMA_COLUMNS = {
    "particle_id",
    "time_s",
    "electron_density_per_m3",
    "positive_ion_density_per_m3",
    "positive_ion_mass_kg",
    "ion_velocity_r_m_per_s",
    "ion_velocity_z_m_per_s",
    "ion_thermal_energy_eV_as_V",
    "charging_current_scale_per_s",
    "screening_length_m",
    "ion_neutral_mean_free_path_m"
  };

  private static final String[] PLASMA_UNITS = {
    "1", "s", "1/m^3", "1/m^3", "kg", "m/s", "m/s", "V", "1/s", "m", "m"
  };

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C0B|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static String stepExpression(int code) {
    if (code == 3125) return "0.3125[us]";
    if (code == 15625) return "0.15625[us]";
    return code + "[ns]";
  }

  private static String stepDirectory(int code) {
    if (code == 625) return "dt_0p625us";
    if (code == 3125) return "dt_0p3125us";
    return "dt_0p15625us";
  }

  private static String stepSeconds(int code) {
    if (code == 625) return "6.25e-7";
    if (code == 3125) return "3.125e-7";
    return "1.5625e-7";
  }

  private static void configurePhysics(Model model, String study) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    for (String tag : new String[] {
        "bf1", "lf1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1"
    }) {
      require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
    }
    physics.feature("bf1").active(false);
    physics.feature("lf1").active(false);
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1"
    }) physics.feature(tag).active(true);
    require(!physics.feature("bf1").isActive(), "Brownian feature remained active");
    require(!physics.feature("lf1").isActive(), "Saffman feature remained active");
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1",
        "relg1", "wall1", "outin", "outpump", "axi1"
    }) require(physics.feature(tag).isActive(), "Required feature is inactive: " + tag);
    require(
        physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage must remain enabled");
    require(
        !physics.prop("StoreExtra").getBoolean("StoreExtra"),
        "Pilot must retain the source model's 121-frame StoreExtra=false contract");
    require(
        "1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Pilot must retain source WallAccuracyOrder=1");
    physics.feature("auxq").set("StudyStep", study + "/time");
    model.param().set("d0", "100[nm]");
  }

  private static void allOff(StudyFeature step, Model model) {
    for (String tag : model.component("comp1").physics().tags()) {
      try {
        step.setSolveFor("/physics/" + tag, false);
      } catch (Throwable ignored) {
      }
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      try {
        step.setSolveFor("/multiphysics/" + tag, false);
      } catch (Throwable ignored) {
      }
    }
  }

  private static String baseSolver(Model model, String study) {
    String[] direct = model.study(study).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(study).getSolverSequences("All")) {
      try {
        if (study.equals(model.sol(tag).study())) return tag;
      } catch (Throwable ignored) {
      }
    }
    throw new IllegalStateException("No solver sequence for " + study);
  }

  private static String resultStore(Model model, String study, String base) {
    for (String kind : new String[] {"ParametricStore", "Parametric"}) {
      String[] values = model.study(study).getSolverSequences(kind);
      if (values.length > 0) return values[0];
    }
    return base;
  }

  private static String createAndRunStudy(Model model, int stepCode) {
    String study = "stdM3C0b" + stepCode;
    String step = stepExpression(stepCode);
    model.param().set("M3C0b_dt", step);
    require(!has(model.study().tags(), study), "Pilot study tag already exists: " + study);
    model.study().create(study);
    model.study(study).label("M3-C0b Brownian-off full deterministic Case A 100 nm");

    model.study(study).create("param", "Parametric");
    StudyFeature parameter = model.study(study).feature("param");
    parameter.set("pname", new String[] {"d0"});
    parameter.set("plistarr", new String[] {"100"});
    parameter.set("punit", new String[] {"nm"});
    parameter.set("sweeptype", "sparse");
    try {
      parameter.set("keepsol", "all");
    } catch (Throwable ignored) {
    }

    model.study(study).create("time", "Transient");
    StudyFeature time = model.study(study).feature("time");
    allOff(time, model);
    time.setSolveFor("/physics/" + PHYSICS, true);
    time.set("tlist", TLIST);
    time.set("usertol", true);
    time.set("rtol", "1e-8");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", BACKGROUND_STUDY);
    time.set("notstudystep", "stat");
    time.set("notsol", BACKGROUND_SOLUTION);
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");
    time.label("Particle-only from saved sol26; fixed classical RK4 " + step);

    configurePhysics(model, study);
    model.study(study).createAutoSequences("all");
    String base = baseSolver(model, study);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3C0b_dt");
    transientSolver.set("rtol", "1e-8");

    long started = System.nanoTime();
    model.study(study).run();
    emit(
        "solve_pass",
        "step_s",
        stepSeconds(stepCode),
        "seconds",
        String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base",
        base);
    return resultStore(model, study, base);
  }

  private static void createParticleDataset(Model model, String dataset, String solution) {
    model.result().dataset().create(dataset, "Particle");
    DatasetFeature data = model.result().dataset(dataset);
    data.set("solution", solution);
    data.set("posdof", new String[] {"comp1.q3r", "comp1.q3z"});
    data.set("geom", "geom1");
    data.set("pgeom", "pgeom_fptas");
    data.set("pgeomspec", "fromphysics");
    data.set("physicsinterface", PHYSICS);
  }

  private static String[] customForce(Physics physics, String feature) {
    String[] force = physics.feature(feature).getStringArray("F");
    require(force.length >= 3, "Expected a three-component force for " + feature);
    return force;
  }

  private static String chargeRate(Physics physics) {
    String[] rate = physics.feature("auxq").getStringArray("R");
    require(rate.length >= 1, "Missing dynamic-charge rate expression");
    return rate[0];
  }

  private static String[] stateExpressions(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    String[] values = {
      "fptas.pidx",
      "t",
      "q3r",
      "q3z",
      "fptas.vr",
      "fptas.vz",
      "ZAS",
      "particlestatus",
      "fptas.fs",
      "fptas.st",
      chargeRate(physics),
      "rho_p*pi*d0^3/6"
    };
    require(values.length == STATE_COLUMNS.length, "State expression/column count mismatch");
    require(STATE_UNITS.length == STATE_COLUMNS.length, "State unit/column count mismatch");
    emit("formula", "feature", "auxq", "R", chargeRate(physics));
    return values;
  }

  private static String[] forceExpressions(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    String[] ion = customForce(physics, "idf");
    String[] lift = customForce(physics, "liftfm");
    String[] dep = customForce(physics, "depf");
    String[] values = {
      "fptas.pidx",
      "t",
      "fptas.ef1.Fer",
      "fptas.ef1.Fez",
      ion[0],
      ion[2],
      "fptas.df1.FDr",
      "fptas.df1.FDz",
      "fptas.thpf1.Ftfr",
      "fptas.thpf1.Ftfz",
      lift[0],
      lift[2],
      dep[0],
      dep[2],
      "fptas.gf1.Fgr",
      "fptas.gf1.Fgz"
    };
    require(values.length == FORCE_COLUMNS.length, "Force expression/column count mismatch");
    require(FORCE_UNITS.length == FORCE_COLUMNS.length, "Force unit/column count mismatch");
    emit("formula", "feature", "idf", "F", Arrays.toString(ion));
    emit("formula", "feature", "liftfm", "F", Arrays.toString(lift));
    emit("formula", "feature", "depf", "F", Arrays.toString(dep));
    return values;
  }

  private static String[] neutralExpressions() {
    String[] values = {
      "fptas.pidx",
      "t",
      "root.comp1.AS_ugr",
      "root.comp1.AS_ugz",
      "root.comp1.AS_Tg",
      "root.comp1.AS_rhog",
      "root.comp1.AS_mug",
      "k_mix",
      "root.comp1.AS_lambdag",
      "d(root.comp1.AS_Tg,r)",
      "d(root.comp1.AS_Tg,z)",
      "-k_mix*d(root.comp1.AS_Tg,r)",
      "-k_mix*d(root.comp1.AS_Tg,z)",
      "d(root.comp1.AS_ugr,z)-d(root.comp1.AS_ugz,r)",
      "d0"
    };
    require(values.length == NEUTRAL_COLUMNS.length, "Neutral expression/column count mismatch");
    require(NEUTRAL_UNITS.length == NEUTRAL_COLUMNS.length, "Neutral unit/column count mismatch");
    return values;
  }

  private static String[] electricExpressions() {
    String[] values = {
      "fptas.pidx",
      "t",
      "root.comp1.AS_Er",
      "root.comp1.AS_Ez",
      "root.comp1.AS_E2",
      "d(root.comp1.AS_E2,r)",
      "d(root.comp1.AS_E2,z)",
      "root.comp1.AS_phi1"
    };
    require(values.length == ELECTRIC_COLUMNS.length, "Electric expression/column count mismatch");
    require(ELECTRIC_UNITS.length == ELECTRIC_COLUMNS.length, "Electric unit/column count mismatch");
    return values;
  }

  private static String[] plasmaExpressions() {
    String[] values = {
      "fptas.pidx",
      "t",
      "root.comp1.AS_ne",
      "root.comp1.AS_ni",
      "AS_mi",
      "root.comp1.AS_uir",
      "root.comp1.AS_uiz",
      "root.comp1.AS_TiV",
      "root.comp1.AS_Ge0",
      "root.comp1.AS_lambdaD",
      "root.comp1.AS_lambda_in"
    };
    require(values.length == PLASMA_COLUMNS.length, "Plasma expression/column count mismatch");
    require(PLASMA_UNITS.length == PLASMA_COLUMNS.length, "Plasma unit/column count mismatch");
    return values;
  }

  private static void exportTable(
      Model model,
      String dataset,
      String directory,
      String tag,
      String filename,
      String[] values,
      String[] units,
      String[] columns) {
    model.result().export().create(tag, "Data");
    ExportFeature export = model.result().export(tag);
    export.set("data", dataset);
    export.set("expr", values);
    export.set("unit", units);
    export.set("descr", columns);
    export.set("filename", directory + "/" + filename);
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "all");
    try {
      export.set("outerinput", "all");
    } catch (Throwable ignored) {
    }
    emit("export_start", "table", filename);
    export.run();
    emit("export_pass", "table", filename);
  }

  private static void exportHistory(Model model, String dataset, String directory) {
    exportTable(
        model,
        dataset,
        directory,
        "m3c0bState",
        "state_raw_wide.csv",
        stateExpressions(model),
        STATE_UNITS,
        STATE_COLUMNS);
    exportTable(
        model,
        dataset,
        directory,
        "m3c0bForce",
        "force_raw_wide.csv",
        forceExpressions(model),
        FORCE_UNITS,
        FORCE_COLUMNS);
    exportTable(
        model,
        dataset,
        directory,
        "m3c0bNeutral",
        "neutral_raw_wide.csv",
        neutralExpressions(),
        NEUTRAL_UNITS,
        NEUTRAL_COLUMNS);
    exportTable(
        model,
        dataset,
        directory,
        "m3c0bElectric",
        "electric_raw_wide.csv",
        electricExpressions(),
        ELECTRIC_UNITS,
        ELECTRIC_COLUMNS);
    exportTable(
        model,
        dataset,
        directory,
        "m3c0bPlasma",
        "plasma_raw_wide.csv",
        plasmaExpressions(),
        PLASMA_UNITS,
        PLASMA_COLUMNS);
  }

  private static int particleRows(Model model, String dataset) {
    String tag = "m3c0bParticleCount";
    try {
      model.result().numerical().create(tag, "Particle");
      NumericalFeature numerical = model.result().numerical(tag);
      numerical.set("data", dataset);
      numerical.set("expr", "fptas.pidx");
      numerical.set("unit", "1");
      numerical.set("innerinput", "first");
      int count = 0;
      for (double[] row : numerical.getReal(false)) count += row.length;
      return count;
    } finally {
      try {
        model.result().numerical().remove(tag);
      } catch (Throwable ignored) {
      }
    }
  }

  private static void runOne(int stepCode) throws Exception {
    Model model = null;
    String directory = stepDirectory(stepCode);
    try {
      model = ModelUtil.loadCopy("M3C0bPilot" + stepCode, SOURCE);
      String solution = createAndRunStudy(model, stepCode);
      String dataset = "partM3C0b" + stepCode;
      createParticleDataset(model, dataset, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == 46, "Expected 46 output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 4.5e-4) < 1e-14, "Unexpected final time");
      int particles = particleRows(model, dataset);
      require(particles == 287, "Expected 287 particles, got " + particles);
      exportHistory(model, dataset, directory);
      emit(
          "configuration",
          "step_s",
          stepSeconds(stepCode),
          "brownian_active",
          "false",
          "saffman_active",
          "false",
          "dynamic_charge_active",
          "true",
          "store_particle_status",
          "true",
          "store_extra",
          "false",
          "wall_accuracy_order",
          "1",
          "integrator",
          "classical_rk4",
          "integrator_order",
          "4",
          "relative_tolerance",
          "1e-8",
          "background_study",
          BACKGROUND_STUDY,
          "background_solution",
          BACKGROUND_SOLUTION,
          "deterministic_contributions",
          "electric,relative_flow_ion_drag,epstein_drag,waldmann_thermophoresis,free_molecular_lift_sensitivity,dielectrophoresis,gravity_buoyancy",
          "physics",
          PHYSICS,
          "study",
          "stdM3C0b" + stepCode,
          "solution",
          solution,
          "output_times",
          Integer.toString(times.length),
          "particle_rows",
          Integer.toString(particles),
          "source_model",
          SOURCE,
          "model_saved",
          "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(false);
    for (int stepCode : STEP_CODES) runOne(stepCode);
    emit(
        "run_pass",
        "case",
        "caseA_100nm",
        "ion_drag_revision",
        "relative_flow_screened_collection_orbital_v1",
        "steps",
        "6.25e-7,3.125e-7,1.5625e-7",
        "model_saved",
        "false");
  }
}

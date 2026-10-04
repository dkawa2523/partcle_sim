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
import java.util.Locale;

/**
 * Runs isolated, force-free positive examples for the COMSOL boundary semantics
 * used by the reference chamber model.
 *
 * <p>Each scenario starts one particle on a normal, analytically known path to
 * exactly one selected terminal boundary. The source MPH is loaded through
 * {@code loadCopy} for every scenario and step and is never saved. This is an
 * external V&V probe; it does not implement or modify production solver laws.
 */
public final class RunM3CBoundarySemantics {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = "fptas";
  private static final String BACKGROUND_STUDY = "stdASf";
  private static final String BACKGROUND_SOLUTION = "sol26";
  private static final String TLIST = "range(0[s],2.5[us],150[us])";
  private static final int[] STEP_CODES = {10000, 5000, 2500};

  private static final String[] STATE_COLUMNS = {
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s"
  };

  private static final String[] STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1", "s"
  };

  // Keep this entry point as one class file. COMSOL's batch method loader does
  // not load companion class files produced for Java nested classes.
  private static final String[] SCENARIO_IDS = {
    "freeze_inlet_37", "disappear_pump_35"
  };
  private static final String[] RELEASE_R = {"23.927", "21.0"};
  private static final String[] RELEASE_Z = {"11.5", "0.073"};
  private static final String[] VELOCITY_R = {"10[m/s]", "0[m/s]"};
  private static final String[] VELOCITY_Z = {"0[m/s]", "-10[m/s]"};
  private static final String[] BOUNDARY_FEATURES = {"outin", "outpump"};
  private static final int[] BOUNDARY_IDS = {37, 35};
  private static final String[] WALL_CONDITIONS = {"Freeze", "Disappear"};
  private static final int[] EXPECTED_STATUSES = {2, 4};

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static boolean contains(int[] values, int target) {
    for (int value : values) if (value == target) return true;
    return false;
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3CB|" + type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static String stepExpression(int code) {
    if (code == 10000) return "10[us]";
    if (code == 5000) return "5[us]";
    return "2.5[us]";
  }

  private static String stepDirectory(int code) {
    if (code == 10000) return "dt_10us";
    if (code == 5000) return "dt_5us";
    return "dt_2p5us";
  }

  private static String stepSeconds(int code) {
    if (code == 10000) return "1e-5";
    if (code == 5000) return "5e-6";
    return "2.5e-6";
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

  private static void configurePhysics(Model model, int scenario, String study) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    for (String tag : new String[] {
        "pp1", "relg1", "wall1", "outin", "outpump", "axi1",
        "bf1", "lf1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1"
    }) require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);

    for (String tag : new String[] {
        "bf1", "lf1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf", "gf1"
    }) physics.feature(tag).active(false);
    for (String tag : new String[] {"pp1", "relg1", "wall1", "outin", "outpump", "axi1"}) {
      physics.feature(tag).active(true);
    }

    model.param().set("d0", "100[nm]");
    PhysicsFeature particle = physics.feature("pp1");
    particle.set("ChargeSpecification", "UserDefined");
    particle.set("Z", "0");

    PhysicsFeature release = physics.feature("relg1");
    release.set("GridType", "AllCombinations");
    release.set("x0", new String[] {RELEASE_R[scenario], RELEASE_Z[scenario]});
    release.set(
        "v0", new String[] {VELOCITY_R[scenario], "0[m/s]", VELOCITY_Z[scenario]});
    release.set("Nvel", 1);
    release.set("Nvalt", 1);
    release.set("StudyStep", study + "/time");

    physics.feature("wall1").set("WallCondition", "Stick");
    physics.feature("outin").set("WallCondition", "Freeze");
    physics.feature("outpump").set("WallCondition", "Disappear");
    PhysicsFeature terminal = physics.feature(BOUNDARY_FEATURES[scenario]);
    require(
        contains(terminal.selection().entities(), BOUNDARY_IDS[scenario]),
        "Target boundary is absent from " + BOUNDARY_FEATURES[scenario]);
    require(
        WALL_CONDITIONS[scenario].equals(terminal.getString("WallCondition")),
        "Target wall condition differs from the locked scenario");

    require(
        physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage must remain enabled");
    require(
        !physics.prop("StoreExtra").getBoolean("StoreExtra"),
        "Boundary probe must retain StoreExtra=false");
    require(
        "1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Boundary probe must retain WallAccuracyOrder=1");
  }

  private static String createAndRunStudy(Model model, int scenario, int stepCode) {
    String study = "stdM3CB" + EXPECTED_STATUSES[scenario] + stepCode;
    String step = stepExpression(stepCode);
    model.param().set("M3CB_dt", step);
    require(!has(model.study().tags(), study), "Boundary study tag already exists: " + study);
    model.study().create(study);
    model.study(study).label("M3-C boundary semantics " + SCENARIO_IDS[scenario]);

    model.study(study).create("param", "Parametric");
    StudyFeature parameter = model.study(study).feature("param");
    parameter.set("pname", new String[] {"d0"});
    parameter.set("plistarr", new String[] {"100"});
    parameter.set("punit", new String[] {"nm"});
    parameter.set("sweeptype", "sparse");

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
    time.label("Force-free boundary probe; fixed classical RK4 " + step);

    configurePhysics(model, scenario, study);
    model.study(study).createAutoSequences("all");
    String base = baseSolver(model, study);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3CB_dt");
    transientSolver.set("rtol", "1e-8");

    long started = System.nanoTime();
    model.study(study).run();
    emit(
        "solve_pass",
        "scenario", SCENARIO_IDS[scenario],
        "step_s", stepSeconds(stepCode),
        "seconds", String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base", base);
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

  private static int particleRows(Model model, String dataset) {
    String tag = "m3cbParticleCount";
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

  private static void exportHistory(Model model, String dataset, String directory) {
    String[] expressions = {
      "fptas.pidx",
      "t",
      "q3r",
      "q3z",
      "fptas.vr",
      "fptas.vz",
      "particlestatus",
      "fptas.fs",
      "fptas.st"
    };
    require(expressions.length == STATE_COLUMNS.length, "State expression count mismatch");
    model.result().export().create("m3cbState", "Data");
    ExportFeature export = model.result().export("m3cbState");
    export.set("data", dataset);
    export.set("expr", expressions);
    export.set("unit", STATE_UNITS);
    export.set("descr", STATE_COLUMNS);
    export.set("filename", directory + "/state_raw_wide.csv");
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
    export.run();
  }

  private static void runOne(int scenario, int stepCode) throws Exception {
    Model model = null;
    String directory = SCENARIO_IDS[scenario] + "/" + stepDirectory(stepCode);
    try {
      emit(
          "load_start",
          "scenario", SCENARIO_IDS[scenario],
          "step_s", stepSeconds(stepCode),
          "source_model", SOURCE);
      model = ModelUtil.loadCopy("M3CB" + EXPECTED_STATUSES[scenario] + stepCode, SOURCE);
      emit(
          "load_pass",
          "scenario", SCENARIO_IDS[scenario],
          "step_s", stepSeconds(stepCode));
      String solution = createAndRunStudy(model, scenario, stepCode);
      String dataset = "partM3CB" + EXPECTED_STATUSES[scenario] + stepCode;
      createParticleDataset(model, dataset, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == 61, "Expected 61 output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 1.5e-4) < 1e-14, "Unexpected final time");
      int particles = particleRows(model, dataset);
      require(particles == 1, "Expected one particle, got " + particles);
      exportHistory(model, dataset, directory);
      emit(
          "configuration",
          "scenario", SCENARIO_IDS[scenario],
          "boundary_feature", BOUNDARY_FEATURES[scenario],
          "boundary_id", Integer.toString(BOUNDARY_IDS[scenario]),
          "wall_condition", WALL_CONDITIONS[scenario],
          "expected_status", Integer.toString(EXPECTED_STATUSES[scenario]),
          "step_s", stepSeconds(stepCode),
          "force_free", "true",
          "dynamic_charge_active", "false",
          "integrator", "classical_rk4",
          "output_times", Integer.toString(times.length),
          "particle_rows", Integer.toString(particles),
          "source_model", SOURCE,
          "model_saved", "false");
    } catch (Throwable error) {
      error.printStackTrace(System.out);
      emit(
          "run_error",
          "scenario", SCENARIO_IDS[scenario],
          "step_s", stepSeconds(stepCode),
          "exception", error.getClass().getName(),
          "message", String.valueOf(error.getMessage()));
      throw new RuntimeException(error);
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(false);
    emit("run_start", "scenario_count", Integer.toString(SCENARIO_IDS.length));
    for (int scenario = 0; scenario < SCENARIO_IDS.length; scenario++) {
      for (int stepCode : STEP_CODES) runOne(scenario, stepCode);
    }
    emit(
        "run_pass",
        "scenarios", "freeze_inlet_37,disappear_pump_35",
        "steps", "1e-5,5e-6,2.5e-6",
        "model_saved", "false");
  }
}

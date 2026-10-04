import com.comsol.model.DatasetFeature;
import com.comsol.model.ExportFeature;
import com.comsol.model.Model;
import com.comsol.model.NumericalFeature;
import com.comsol.model.SolverFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;
import java.util.Locale;

/**
 * Builds and runs the force-free 2-D axisymmetric critical-boundary microcase.
 *
 * <p>The model is created from scratch. Three particles share one rectangular
 * geometry and one particle interface: one departs from the lower surface, one
 * reflects specularly from the outer radial wall, and one crosses the symmetry
 * axis. No production-solver code or source MPH is loaded.
 */
public final class RunM3CCriticalBoundaries {
  private static final String PHYSICS = "fpt";
  private static final String STUDY = "std1";
  private static final String TLIST = "range(0[s],0.25[ms],6[ms])";
  private static final int[] STEP_CODES = {1000, 500, 250};

  private static final String[] STATE_COLUMNS = {
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "current_status_code",
    "final_status_code"
  };

  private static final String[] STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1"
  };

  private static final String[] RELEASE_TAGS = {"surface", "reflect", "axis"};
  private static final String[] RELEASE_R = {"6[mm]", "8.05[mm]", "1.95[mm]"};
  private static final String[] RELEASE_Z = {"0[mm]", "6[mm]", "14[mm]"};
  private static final String[] VELOCITY_R = {"0[m/s]", "1[m/s]", "-1[m/s]"};
  private static final String[] VELOCITY_Z = {"1[m/s]", "0.2[m/s]", "0.1[m/s]"};

  private RunM3CCriticalBoundaries() {}

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
    StringBuilder line = new StringBuilder("M3CCB|" + type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static String stepExpression(int code) {
    if (code == 1000) return "1[ms]";
    if (code == 500) return "0.5[ms]";
    return "0.25[ms]";
  }

  private static String stepDirectory(int code) {
    if (code == 1000) return "dt_1ms";
    if (code == 500) return "dt_0p5ms";
    return "dt_0p25ms";
  }

  private static String stepSeconds(int code) {
    if (code == 1000) return "1e-3";
    if (code == 500) return "5e-4";
    return "2.5e-4";
  }

  private static String baseSolver(Model model) {
    String[] direct = model.study(STUDY).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(STUDY).getSolverSequences("All")) {
      try {
        if (STUDY.equals(model.sol(tag).study())) return tag;
      } catch (Throwable ignored) {
      }
    }
    throw new IllegalStateException("No solver sequence for " + STUDY);
  }

  private static String resultStore(Model model, String base) {
    for (String kind : new String[] {"ParametricStore", "Parametric"}) {
      String[] values = model.study(STUDY).getSolverSequences(kind);
      if (values.length > 0) return values[0];
    }
    return base;
  }

  private static void addRelease(
      Physics physics, String featureTag, int releaseIndex) {
    physics.create(featureTag, "ReleaseGrid", -1);
    PhysicsFeature release = physics.feature(featureTag);
    release.label("M3-C critical boundary " + RELEASE_TAGS[releaseIndex]);
    release.set("GridType", "AllCombinations");
    release.set("x0", new String[] {RELEASE_R[releaseIndex], RELEASE_Z[releaseIndex]});
    release.set(
        "v0",
        new String[] {VELOCITY_R[releaseIndex], "0[m/s]", VELOCITY_Z[releaseIndex]});
    release.set("Nvel", 1);
    release.set("Nvalt", 1);
    release.set("StudyStep", STUDY + "/time");
  }

  private static Model buildModel(int stepCode) {
    Model model = ModelUtil.create("M3CCB" + stepCode);
    model.label("M3-C critical 2-D axisymmetric boundaries");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.component("comp1").geom("geom1").axisymmetric(true);
    model.component("comp1").geom("geom1").lengthUnit("m");
    model.component("comp1").geom("geom1").create("rect1", "Rectangle");
    model.component("comp1").geom("geom1").feature("rect1")
        .set("size", new String[] {"10[mm]", "20[mm]"});
    model.component("comp1").geom("geom1").run();

    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").autoMeshSize(4);
    model.component("comp1").mesh("mesh1").run();

    model.component("comp1").physics().create(PHYSICS, "FluidParticleTracing", "geom1");
    Physics physics = model.component("comp1").physics(PHYSICS);
    require(has(physics.feature().tags(), "pp1"), "Missing default particle properties");
    require(has(physics.feature().tags(), "wall1"), "Missing default wall feature");
    require(has(physics.feature().tags(), "axi1"), "Missing axis-symmetry feature");

    physics.prop("Formulation").setIndex("Formulation", "NewtonianFirstOrder", 0);
    physics.prop("StoreParticleStatusData").set("StoreParticleStatusData", true);
    physics.prop("StoreExtra").set("StoreExtra", false);
    physics.prop("WallAccuracyOrder").set("WallAccuracyOrder", "1");
    physics.feature("pp1").set("rhop_mat", "userdef");
    physics.feature("pp1").set("rhop", "1000[kg/m^3]");
    physics.feature("pp1").set("dp", "100[nm]");
    physics.feature("wall1").set("WallCondition", "Bounce");
    physics.feature("axi1").set("WallCondition", "Bounce");

    int[] axisBoundaries = physics.feature("axi1").selection().entities();
    int[] wallBoundaries = physics.feature("wall1").selection().entities();
    require(axisBoundaries.length == 1, "Expected one symmetry-axis boundary");
    require(contains(axisBoundaries, 1), "Rectangle axis must be boundary 1");
    require(!contains(wallBoundaries, 1), "Axis boundary must not be a wall");
    require(contains(wallBoundaries, 2), "Lower release surface must use the wall law");
    require(contains(wallBoundaries, 4), "Outer radial boundary must use the wall law");

    model.study().create(STUDY);
    model.study(STUDY).create("time", "Transient");
    model.study(STUDY).feature("time").set("tlist", TLIST);
    model.study(STUDY).feature("time").set("usertol", true);
    model.study(STUDY).feature("time").set("rtol", "1e-9");

    addRelease(physics, "relgSurface", 0);
    addRelease(physics, "relgReflect", 1);
    addRelease(physics, "relgAxis", 2);

    model.study(STUDY).createAutoSequences("all");
    String base = baseSolver(model);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", stepExpression(stepCode));
    transientSolver.set("rtol", "1e-9");

    emit(
        "configuration",
        "step_s", stepSeconds(stepCode),
        "geometry", "axisymmetric_rectangle_10mm_by_20mm",
        "axis_boundary", Arrays.toString(axisBoundaries),
        "axis_feature_type", physics.feature("axi1").getType(),
        "axis_condition", physics.feature("axi1").getString("WallCondition"),
        "wall_boundaries", Arrays.toString(wallBoundaries),
        "wall_condition", physics.feature("wall1").getString("WallCondition"),
        "release_groups", String.join(",", RELEASE_TAGS),
        "force_free", "true",
        "dynamic_charge_active", "false",
        "integrator", "classical_rk4",
        "model_source", "from_scratch",
        "model_saved", "false");
    return model;
  }

  private static void createParticleDataset(Model model, String solution) {
    model.result().dataset().create("particles", "Particle");
    DatasetFeature data = model.result().dataset("particles");
    data.set("solution", solution);
    data.set("posdof", new String[] {"comp1.qr", "comp1.qz"});
    data.set("geom", "geom1");
    data.set("pgeom", "pgeom_fpt");
    data.set("pgeomspec", "fromphysics");
    data.set("physicsinterface", PHYSICS);
  }

  private static int particleRows(Model model) {
    String tag = "particleCount";
    try {
      model.result().numerical().create(tag, "Particle");
      NumericalFeature numerical = model.result().numerical(tag);
      numerical.set("data", "particles");
      numerical.set("expr", "fpt.pidx");
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

  private static void exportHistory(Model model, String directory) {
    String[] expressions = {
      "fpt.pidx", "t", "qr", "qz", "fpt.vr", "fpt.vz", "particlestatus", "fpt.fs"
    };
    require(expressions.length == STATE_COLUMNS.length, "State expression count mismatch");
    model.result().export().create("state", "Data");
    ExportFeature export = model.result().export("state");
    export.set("data", "particles");
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
    export.run();
  }

  private static void runOne(int stepCode) throws Exception {
    Model model = null;
    try {
      model = buildModel(stepCode);
      long started = System.nanoTime();
      model.study(STUDY).run();
      String base = baseSolver(model);
      String solution = resultStore(model, base);
      createParticleDataset(model, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == 25, "Expected 25 output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 6e-3) < 1e-13, "Unexpected final time");
      int particles = particleRows(model);
      require(particles == 3, "Expected three particles, got " + particles);
      exportHistory(model, stepDirectory(stepCode));
      emit(
          "solve_pass",
          "step_s", stepSeconds(stepCode),
          "seconds", String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
          "output_times", Integer.toString(times.length),
          "particle_rows", Integer.toString(particles));
    } catch (Throwable error) {
      error.printStackTrace(System.out);
      emit(
          "run_error",
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
    emit("run_start", "step_count", Integer.toString(STEP_CODES.length));
    for (int stepCode : STEP_CODES) runOne(stepCode);
    emit(
        "run_pass",
        "particles", "surface,reflect,axis",
        "steps", "1e-3,5e-4,2.5e-4",
        "model_source", "from_scratch",
        "model_saved", "false");
  }
}

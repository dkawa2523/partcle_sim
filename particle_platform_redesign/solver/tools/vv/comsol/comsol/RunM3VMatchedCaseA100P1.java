import com.comsol.model.DatasetFeature;
import com.comsol.model.ExportFeature;
import com.comsol.model.FunctionFeature;
import com.comsol.model.Model;
import com.comsol.model.SolverFeature;
import com.comsol.model.StudyFeature;
import com.comsol.model.physics.Physics;
import com.comsol.model.util.ModelUtil;
import java.util.Arrays;
import java.util.Locale;

/** Runs a COMSOL companion using the candidate's canonical P1 node tables. */
public final class RunM3VMatchedCaseA100P1 {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = "fptas";
  private static final String STUDY = "stdM3VP1";
  private static final String DATASET = "partM3VP1";
  private static final String TLIST = "range(0[s],1e-5[s],4e-4[s])";
  private static final int[] STEP_US = {10, 5, 25};
  private static final String[] NAMES = {
      "m3v_ugr", "m3v_ugz", "m3v_Tg", "m3v_rhog", "m3v_mug", "m3v_Er", "m3v_Ez"
  };
  private static final String[] UNITS = {
      "m/s", "m/s", "K", "kg/m^3", "Pa*s", "V/m", "V/m"
  };

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3V_MATCHED_P1|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static void createFunctions(Model model) {
    for (int index = 0; index < NAMES.length; index++) {
      String tag = "m3vF" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", NAMES[index] + "_sectionwise.txt");
      function.set("struct", "sectionwise");
      function.set("funcs", new String[][] {{NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", UNITS[index]);
      function.importData();
    }
  }

  private static void deactivate(Physics physics, String tag) {
    require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
    physics.feature(tag).active(false);
  }

  private static void configurePhysics(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    for (String tag : new String[] {
        "bf1", "auxq", "idf", "thpf1", "lf1", "liftfm", "depf"
    }) deactivate(physics, tag);
    physics.feature("pp1").set("ChargeSpecification", "UserDefined");
    physics.feature("pp1").set("Z", "-1");
    physics.feature("ef1").set(
        "E", new String[] {"m3v_Er(r,z)", "0[V/m]", "m3v_Ez(r,z)"});
    physics.feature("df1").set(
        "u", new String[] {"m3v_ugr(r,z)", "0[m/s]", "m3v_ugz(r,z)"});
    physics.feature("df1").set("rho", "m3v_rhog(r,z)");
    physics.feature("df1").set("mu", "m3v_mug(r,z)");
    physics.feature("df1").set("minput_temperature", "m3v_Tg(r,z)");
    // COMSOL's ideal-gas Epstein implementation uses p/rho as well as T.
    // Reconstruct pressure from the same interpolated rho/T and the audited
    // 80/20 CF4/O2 mixture mass so the common force law has one field owner.
    physics.feature("df1").set(
        "pA", "m3v_rhog(r,z)*k_B_const*m3v_Tg(r,z)/(Mmix/N_A_const)");
    physics.feature("gf1").set("rho", "m3v_rhog(r,z)");
    for (String tag : new String[] {"ef1", "df1", "gf1"}) {
      require(has(physics.feature().tags(), tag), "Missing retained feature " + tag);
      physics.feature(tag).active(true);
    }
    model.param().set("d0", "100[nm]");
  }

  private static void allOff(StudyFeature step, Model model) {
    for (String tag : model.component("comp1").physics().tags()) {
      try { step.setSolveFor("/physics/" + tag, false); }
      catch (Throwable ignored) {}
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      try { step.setSolveFor("/multiphysics/" + tag, false); }
      catch (Throwable ignored) {}
    }
  }

  private static String baseSolver(Model model) {
    String[] solver = model.study(STUDY).getSolverSequences("SolverSequence");
    if (solver.length > 0) return solver[0];
    for (String tag : model.study(STUDY).getSolverSequences("All")) {
      try { if (STUDY.equals(model.sol(tag).study())) return tag; }
      catch (Throwable ignored) {}
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

  private static String createAndRunStudy(Model model, String step) {
    model.param().set("M3V_dt", step);
    model.study().create(STUDY);
    model.study(STUDY).label("M3-V canonical-P1 common-physics Case A 100 nm");
    model.study(STUDY).create("param", "Parametric");
    StudyFeature parameter = model.study(STUDY).feature("param");
    parameter.set("pname", new String[] {"d0"});
    parameter.set("plistarr", new String[] {"100"});
    parameter.set("punit", new String[] {"nm"});
    parameter.set("sweeptype", "sparse");
    try { parameter.set("keepsol", "all"); }
    catch (Throwable ignored) {}

    model.study(STUDY).create("time", "Transient");
    StudyFeature time = model.study(STUDY).feature("time");
    allOff(time, model);
    time.setSolveFor("/physics/" + PHYSICS, true);
    time.set("tlist", TLIST);
    time.set("usertol", true);
    time.set("rtol", "1e-8");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", "stdASf");
    time.set("notsol", "sol26");
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");

    model.study(STUDY).createAutoSequences("all");
    String base = baseSolver(model);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3V_dt");
    transientSolver.set("rtol", "1e-8");

    long started = System.nanoTime();
    model.study(STUDY).run();
    emit("solve_pass", "step", step, "seconds",
        String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base", base);
    return resultStore(model, base);
  }

  private static void createParticleDataset(Model model, String solution) {
    model.result().dataset().create(DATASET, "Particle");
    DatasetFeature data = model.result().dataset(DATASET);
    data.set("solution", solution);
    data.set("posdof", new String[] {"comp1.q3r", "comp1.q3z"});
    data.set("geom", "geom1");
    data.set("pgeom", "pgeom_fptas");
    data.set("pgeomspec", "fromphysics");
    data.set("physicsinterface", PHYSICS);
  }

  private static void exportHistory(Model model, String directory) {
    String tag = "m3vP1History";
    model.result().export().create(tag, "Data");
    ExportFeature export = model.result().export(tag);
    export.set("data", DATASET);
    export.set("expr", new String[] {
        "fptas.pidx", "t", "q3r", "q3z", "fptas.vr", "fptas.vz",
        "-1", "particlestatus", "fptas.fs", "fptas.st",
        "fptas.ef1.Fer", "fptas.ef1.Fez", "fptas.df1.FDr", "fptas.df1.FDz",
        "fptas.gf1.Fgr", "fptas.gf1.Fgz",
        "(fptas.ef1.Fer+fptas.df1.FDr+fptas.gf1.Fgr)/(rho_p*pi*d0^3/6)",
        "(fptas.ef1.Fez+fptas.df1.FDz+fptas.gf1.Fgz)/(rho_p*pi*d0^3/6)",
        "m3v_ugr(q3r,q3z)", "m3v_ugz(q3r,q3z)", "m3v_Tg(q3r,q3z)",
        "m3v_rhog(q3r,q3z)", "m3v_mug(q3r,q3z)",
        "m3v_Er(q3r,q3z)", "m3v_Ez(q3r,q3z)", "rho_p*pi*d0^3/6"
    });
    export.set("unit", new String[] {
        "1", "s", "m", "m", "m/s", "m/s", "1", "1", "1", "s",
        "N", "N", "N", "N", "N", "N", "m/s^2", "m/s^2",
        "m/s", "m/s", "K", "kg/m^3", "Pa*s", "V/m", "V/m", "kg"
    });
    export.set("descr", new String[] {
        "particle_id", "time_s", "r_m", "z_m", "velocity_r_m_per_s",
        "velocity_z_m_per_s", "charge_number_e", "current_status_code",
        "final_status_code", "stop_or_event_time_s", "electric_force_r_N",
        "electric_force_z_N", "epstein_force_r_N", "epstein_force_z_N",
        "gravity_buoyancy_force_r_N", "gravity_buoyancy_force_z_N",
        "acceleration_r_m_per_s2", "acceleration_z_m_per_s2",
        "gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s", "gas_temperature_K",
        "gas_density_kg_per_m3", "gas_dynamic_viscosity_Pa_s",
        "electric_field_r_V_per_m", "electric_field_z_V_per_m", "particle_mass_kg"
    });
    export.set("filename", directory + "history_raw_wide.csv");
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "all");
    try { export.set("outerinput", "all"); }
    catch (Throwable ignored) {}
    export.run();
  }

  private static void runOne(String root, int stepCode) throws Exception {
    String step = stepCode == 25 ? "2.5[us]" : stepCode + "[us]";
    String name = stepCode == 25 ? "dt_2p5us" : "dt_" + stepCode + "us";
    Model model = null;
    try {
      model = ModelUtil.loadCopy("M3VMatchedP1" + stepCode, SOURCE);
      createFunctions(model);
      configurePhysics(model);
      String solution = createAndRunStudy(model, step);
      createParticleDataset(model, solution);
      exportHistory(model, root + name + "/");
      double[] times = model.sol(solution).getPVals();
      require(times.length == 41, "Expected 41 output times, got " + times.length);
      emit("export_pass", "step", step, "directory", name, "solution", solution,
          "times", Integer.toString(times.length), "model_saved", "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  public static void main(String[] args) throws Exception {
    ModelUtil.showProgress(false);
    for (int step : STEP_US) runOne("", step);
    emit("run_pass", "source", SOURCE, "model_saved", "false",
        "field_source", "canonical_p1_sectionwise_exact_mesh_linear");
  }
}

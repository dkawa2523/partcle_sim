import com.comsol.model.DatasetFeature;
import com.comsol.model.ExportFeature;
import com.comsol.model.FunctionFeature;
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
 * Runs the deterministic M3-C3 Case-P 100 nm three-current COMSOL reference.
 *
 * <p>The working directory contains an isolated {@code source_copy.mph}, exact-connectivity P1
 * tables, and the shared three-current release table. The source model supplies geometry,
 * boundary selections, and its saved stationary background solution only. It is loaded with
 * {@code loadCopy} and is never saved.
 */
public final class RunM3C3CasePThreeCurrent {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = "fpt";
  private static final String BACKGROUND_STUDY = "std2";
  private static final String BACKGROUND_STEP = "ftper";
  private static final String BACKGROUND_SOLUTION = "sol2";
  private static final String PARTICLE_GEOMETRY = "pgeom_fpt";
  private static final String SOURCE_DATASET = "part_P_100nm";
  private static final String POSITION_R = "qr";
  private static final String POSITION_Z = "qz";
  private static final String CHARGE = "ZP";
  private static final String STUDY = "stdM3C3";
  private static final String EPSTEIN_TAG = "m3c3Epstein";
  private static final String THERMO_TAG = "m3c3HeatFlux";
  private static final String TIME_LIST =
      "range(0[s],1e-5[s],5e-4[s]) "
          + "range(6e-4[s],1e-4[s],5e-3[s]) "
          + "range(6e-3[s],1e-3[s],3e-2[s])";
  private static final String FIXED_STEP_SECONDS = "__M3C3_FIXED_STEP_SECONDS__";
  private static final String DIAGNOSTIC_FIXED_STEP_SECONDS = "1.25e-6";
  private static final int EXPECTED_PARTICLES = 287;
  private static final int EXPECTED_TIMES = 121;

  private static final String[] P1_NAMES = {
    "m3c1_rhog", "m3c1_mug", "m3c1_Tg", "m3c1_lambdag",
    "m3c1_ne", "m3c1_ni", "m3c1_Te", "m3c1_TiV", "m3c1_mi",
    "m3c1_lambdaD", "m3c1_lambdaIn", "m3c1_omegaPhi",
    "m3c1_ugr", "m3c1_ugz", "m3c1_Er", "m3c1_Ez",
    "m3c1_uir", "m3c1_uiz", "m3c1_gradE2r", "m3c1_gradE2z",
    "m3c1_qr", "m3c1_qz",
    "m3c3_nn", "m3c3_unr", "m3c3_unz", "m3c3_TnV", "m3c3_mn"
  };
  private static final String[] P1_UNITS = {
    "kg/m^3", "Pa*s", "K", "m", "1/m^3", "1/m^3", "V", "V", "kg",
    "m", "m", "1/s", "m/s", "m/s", "V/m", "V/m", "m/s", "m/s",
    "V^2/m^3", "V^2/m^3", "W/m^2", "W/m^2",
    "1/m^3", "m/s", "m/s", "V", "kg"
  };
  private static final String[] RELEASE_NAMES = {"m3c1_vr0", "m3c1_vz0", "m3c1_Z0"};
  private static final String[] RELEASE_UNITS = {"m/s", "m/s", "1"};

  private static final String[] STATE_COLUMNS = {
    "particle_id", "time_s", "r_m", "z_m", "velocity_r_m_per_s",
    "velocity_z_m_per_s", "charge_number_e", "current_status_code",
    "final_status_code", "stop_or_event_time_s"
  };
  private static final String[] STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1", "1", "s"
  };

  private static final String[] DIAGNOSTIC_STATE_COLUMNS = {
    "particle_id", "time_s", "r_m", "z_m", "velocity_r_m_per_s",
    "velocity_z_m_per_s", "charge_number_e", "current_status_code",
    "final_status_code", "stop_or_event_time_s"
  };
  private static final String[] DIAGNOSTIC_STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1", "1", "s"
  };

  private static final String[] DIAGNOSTIC_FORCE_COLUMNS = {
    "particle_id", "time_s", "charge_rate_number_s",
    "positive_collection_rate_number_s", "electron_collection_rate_number_s",
    "negative_collection_rate_number_s",
    "electric_force_r_N", "electric_force_z_N",
    "ion_drag_force_r_N", "ion_drag_force_z_N",
    "epstein_drag_force_r_N", "epstein_drag_force_z_N",
    "thermophoretic_force_r_N", "thermophoretic_force_z_N",
    "lift_force_r_N", "lift_force_z_N", "dep_force_r_N", "dep_force_z_N",
    "gravity_buoyancy_force_r_N", "gravity_buoyancy_force_z_N",
    "total_force_r_N", "total_force_z_N"
  };
  private static final String[] DIAGNOSTIC_FORCE_UNITS = {
    "1", "s", "1/s", "1/s", "1/s", "1/s", "N", "N", "N", "N", "N", "N",
    "N", "N", "N", "N", "N", "N", "N", "N", "N", "N"
  };

  private RunM3C3CasePThreeCurrent() {}

  private static String fixedStepSeconds() {
    String value = FIXED_STEP_SECONDS;
    require("1e-5".equals(value) || "5e-6".equals(value) || "2.5e-6".equals(value)
        || "1.25e-6".equals(value),
        "Unsupported M3-C3 fixed RK4 step: " + value);
    return value;
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C3_CASEP|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static void requireEntities(PhysicsFeature feature, int... expected) {
    int[] actual = feature.selection().entities();
    require(Arrays.equals(actual, expected),
        "Unexpected entity selection for " + feature.tag() + ": " + Arrays.toString(actual));
  }

  private static void validateSource(Model model) {
    require(has(model.component("comp1").physics().tags(), PHYSICS),
        "Missing source physics " + PHYSICS);
    require(has(model.study().tags(), BACKGROUND_STUDY),
        "Missing source background study " + BACKGROUND_STUDY);
    require(has(model.sol().tags(), BACKGROUND_SOLUTION),
        "Missing source background solution " + BACKGROUND_SOLUTION);
    require(!model.sol(BACKGROUND_SOLUTION).isEmpty(), "Source background solution is empty");
    require(has(model.result().dataset().tags(), SOURCE_DATASET),
        "Missing source particle dataset " + SOURCE_DATASET);
    DatasetFeature dataset = model.result().dataset(SOURCE_DATASET);
    require(PARTICLE_GEOMETRY.equals(dataset.getString("pgeom")),
        "Particle geometry tag differs");
    require(PHYSICS.equals(dataset.getString("physicsinterface")),
        "Particle dataset physics differs");
    require(Arrays.equals(
        new String[] {"comp1." + POSITION_R, "comp1." + POSITION_Z},
        dataset.getStringArray("posdof")), "Particle position DOFs differ");

    Physics physics = model.component("comp1").physics(PHYSICS);
    for (String tag : new String[] {
        "relg1", "pp1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm",
        "depf", "gf1", "bf1", "lf1", "wall1", "outin", "outpump", "axi1"
    }) require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
    requireEntities(physics.feature("wall1"), 6, 8, 28, 29, 32, 33, 34, 36, 38, 40, 41,
        45, 46, 47);
    requireEntities(physics.feature("outin"), 37);
    requireEntities(physics.feature("outpump"), 35);
    requireEntities(physics.feature("axi1"), 5);
    require("0".equals(physics.prop("IncludeOutOfPlane").getString("IncludeOutOfPlane")),
        "Particle interface must remain R-Z with out-of-plane disabled");
  }

  private static void createFunctions(Model model) {
    require(P1_NAMES.length == P1_UNITS.length, "P1 function metadata mismatch");
    for (int index = 0; index < P1_NAMES.length; index++) {
      String tag = "m3c3P1F" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", P1_NAMES[index] + "_sectionwise.txt");
      function.set("struct", "sectionwise");
      function.set("funcs", new String[][] {{P1_NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.importData();
      // importData resets argument-unit metadata for the imported table.
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", P1_UNITS[index]);
    }
    for (int index = 0; index < RELEASE_NAMES.length; index++) {
      String tag = "m3c3ReleaseF" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", RELEASE_NAMES[index] + ".txt");
      function.set("struct", "spreadsheet");
      function.set("funcs", new String[][] {{RELEASE_NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.importData();
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", RELEASE_UNITS[index]);
    }
  }

  private static String positiveSpeedSquared() {
    return "((m3c1_uir(r,z)-" + PHYSICS + ".vr)^2"
        + "+(m3c1_uiz(r,z)-" + PHYSICS + ".vz)^2"
        + "+8*e_const*m3c1_TiV(r,z)/(pi*m3c1_mi(r,z))+(1[m/s])^2)";
  }

  private static String negativeSpeedSquared() {
    return "((m3c3_unr(r,z)-" + PHYSICS + ".vr)^2"
        + "+(m3c3_unz(r,z)-" + PHYSICS + ".vz)^2"
        + "+8*e_const*m3c3_TnV(r,z)/(pi*m3c3_mn(r,z))+(1[m/s])^2)";
  }

  private static String particlePotential() {
    String screening = "max(d0/2,m3c1_lambdaD(r,z))";
    return "(" + CHARGE + "*e_const/(4*pi*epsilon0_const*(d0/2)"
        + "*(1+(d0/2)/(" + screening + "))))";
  }

  private static String clippedExponential(String argument) {
    return "exp(max(-50,min(50," + argument + ")))";
  }

  private static String[] collectionRates() {
    String potential = particlePotential();
    String positiveSpeed2 = positiveSpeedSquared();
    String positiveEnergy =
        "max(m3c1_mi(r,z)*" + positiveSpeed2 + "/(2*e_const),0.01[V])";
    String positiveFactor = "if(" + potential + "<=0[V],1-" + potential + "/("
        + positiveEnergy + ")," + clippedExponential("-" + potential + "/("
        + positiveEnergy + ")") + ")";
    String electronFactor = "if(" + potential + "<=0[V],"
        + clippedExponential(potential + "/m3c1_Te(r,z)")
        + ",1+" + potential + "/m3c1_Te(r,z))";
    String positive = "pi*(d0/2)^2*m3c1_ni(r,z)*sqrt(" + positiveSpeed2 + ")*("
        + positiveFactor + ")";
    String electron = "pi*(d0/2)^2*m3c1_ne(r,z)"
        + "*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const))*(" + electronFactor + ")";

    String negativeSpeed2 = negativeSpeedSquared();
    String negativeEnergy =
        "max(m3c3_mn(r,z)*" + negativeSpeed2 + "/(2*e_const),0.01[V])";
    String negativeFactor = "if(" + potential + "<=0[V],"
        + clippedExponential(potential + "/(" + negativeEnergy + ")")
        + ",1+" + potential + "/(" + negativeEnergy + "))";
    String negative = "pi*(d0/2)^2*m3c3_nn(r,z)*sqrt(" + negativeSpeed2 + ")*("
        + negativeFactor + ")";
    String total = "(" + positive + "-(" + electron + ")-(" + negative + "))";
    return new String[] {positive, electron, negative, total};
  }

  private static String chargeRate() {
    return collectionRates()[3];
  }

  private static String[] ionDrag() {
    String speed2 = positiveSpeedSquared();
    String potential = particlePotential();
    String screen = "max(d0/2,min(m3c1_lambdaD(r,z),m3c1_lambdaIn(r,z)))";
    String capture = "min((" + screen + ")^2,(d0/2)^2*max(0,1-2*e_const*"
        + potential + "/(m3c1_mi(r,z)*" + speed2 + ")))";
    String b90 = "sqrt(" + CHARGE + "^2+1e-20)*e_const^2/"
        + "(4*pi*epsilon0_const*m3c1_mi(r,z)*" + speed2 + ")";
    String coulombLog = "max(0,0.5*log(((" + screen + ")^2+(" + b90 + ")^2)/("
        + capture + "+(" + b90 + ")^2)))";
    String scalar = "(pi*(" + capture + ")+4*pi*(" + b90 + ")^2*(" + coulombLog + "))"
        + "*m3c1_ni(r,z)*m3c1_mi(r,z)*sqrt(" + speed2 + ")";
    return new String[] {
      scalar + "*(m3c1_uir(r,z)-" + PHYSICS + ".vr)",
      "0[N]",
      scalar + "*(m3c1_uiz(r,z)-" + PHYSICS + ".vz)"
    };
  }

  private static String[] thermophoresis() {
    String denominator =
        "sqrt(8*k_B_const*m3c1_Tg(r,z)/(pi*1.2753471408396638e-25[kg]))";
    return new String[] {
      "(32/15)*(d0/2)^2*m3c1_qr(r,z)/" + denominator,
      "0[N]",
      "(32/15)*(d0/2)^2*m3c1_qz(r,z)/" + denominator
    };
  }

  private static String[] epsteinDrag() {
    return CommonP1Epstein.force(PHYSICS);
  }

  private static String[] lift() {
    String common = "pi*m3c1_rhog(r,z)*m3c1_lambdag(r,z)*(d0/2)^2";
    return new String[] {
      common + "*(m3c1_ugz(r,z)-" + PHYSICS + ".vz)*m3c1_omegaPhi(r,z)",
      "0[N]",
      "-" + common + "*(m3c1_ugr(r,z)-" + PHYSICS + ".vr)*m3c1_omegaPhi(r,z)"
    };
  }

  private static String[] dep() {
    String factor = "2*pi*epsilon0_const*(d0/2)^3*0.5161290322580645";
    return new String[] {
      factor + "*m3c1_gradE2r(r,z)", "0[N]", factor + "*m3c1_gradE2z(r,z)"
    };
  }

  private static void configureForce(
      PhysicsFeature feature, String label, String[] force) {
    feature.label(label);
    feature.selection().set(3);
    feature.set("SpecifyForce", "Directly");
    feature.set("F", force);
    feature.set("ParticlesToAffect", "All");
    feature.set("AffectedParticleProperties", "pp1");
    feature.set("StudyStep", STUDY + "/time");
  }

  private static void configurePhysics(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    physics.feature("bf1").active(false);
    physics.feature("lf1").active(false);
    physics.feature("thpf1").active(false);
    physics.feature("df1").active(false);
    for (String tag : new String[] {"auxq", "idf", "ef1", "liftfm", "depf", "gf1"}) {
      physics.feature(tag).active(true);
    }

    model.param().set("d0", "100[nm]");
    CommonP1Epstein.bind(model);
    physics.feature("relg1").set(
        "v0", new String[] {"m3c1_vr0(r,z)", "0[m/s]", "m3c1_vz0(r,z)"});
    physics.feature("relg1").set("aux0_auxq", "m3c1_Z0(r,z)");
    physics.feature("pp1").set("ChargeSpecification", "UserDefined");
    physics.feature("pp1").set("Z", CHARGE);

    physics.feature("auxq").set("R", new String[] {chargeRate()});
    physics.feature("auxq").set("StudyStep", STUDY + "/time");
    configureForce(physics.feature("idf"), "M3-C3 relative-flow aggregate ion drag", ionDrag());
    physics.feature("ef1").set(
        "E", new String[] {"m3c1_Er(r,z)", "0[V/m]", "m3c1_Ez(r,z)"});
    physics.feature("ef1").set("E_src", "userdef");
    physics.feature("ef1").set("StudyStep", STUDY + "/time");

    require(!has(physics.feature().tags(), EPSTEIN_TAG), "Epstein force tag exists");
    physics.create(EPSTEIN_TAG, "Force", 2);
    configureForce(physics.feature(EPSTEIN_TAG), "M3-C3 explicit linear Epstein drag",
        epsteinDrag());
    configureForce(physics.feature("liftfm"), "M3-C3 rarefied lift", lift());
    configureForce(physics.feature("depf"), "M3-C3 dielectrophoresis", dep());
    require(!has(physics.feature().tags(), THERMO_TAG), "Thermophoresis force tag exists");
    physics.create(THERMO_TAG, "Force", 2);
    configureForce(physics.feature(THERMO_TAG), "M3-C3 Waldmann heat-flux force",
        thermophoresis());
    physics.feature("gf1").set("rho", "m3c1_rhog(r,z)");
    physics.feature("gf1").set("rho_mat", "userdef");
    physics.feature("gf1").set("minput_temperature_src", "userdef");
    physics.feature("gf1").set("minput_temperature", "m3c1_Tg(r,z)");
    physics.feature("gf1").set("StudyStep", STUDY + "/time");

    physics.feature("wall1").set("WallCondition", "Stick");
    physics.feature("outin").set("WallCondition", "Freeze");
    physics.feature("outpump").set("WallCondition", "Disappear");
    physics.feature("axi1").set("WallCondition", "Bounce");
    for (String tag : new String[] {"wall1", "outin", "outpump", "axi1", "relg1", "pp1"}) {
      physics.feature(tag).set("StudyStep", STUDY + "/time");
    }

    require(!physics.feature("bf1").isActive(), "Brownian force remained active");
    require(!physics.feature("lf1").isActive(), "Saffman force remained active");
    require(!physics.feature("thpf1").isActive(), "Native thermophoresis remained active");
    require(!physics.feature("df1").isActive(), "Native drag remained active");
    for (String tag : new String[] {
        "auxq", "idf", "ef1", EPSTEIN_TAG, "liftfm", "depf", "gf1", THERMO_TAG,
        "relg1", "pp1", "wall1", "outin", "outpump", "axi1"
    }) require(physics.feature(tag).isActive(), "Required feature inactive: " + tag);
    require("Stick".equals(physics.feature("wall1").getString("WallCondition")),
        "Material wall is not Stick");
    require("Freeze".equals(physics.feature("outin").getString("WallCondition")),
        "Gas inlet is not Freeze/hold");
    require("Disappear".equals(physics.feature("outpump").getString("WallCondition")),
        "Pump outlet is not Disappear/escape");
    require("Bounce".equals(physics.feature("axi1").getString("WallCondition")),
        "The companion axis must use the registered meridional-fold mapping");
    require(physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage is disabled");
    require(!physics.prop("StoreExtra").getBoolean("StoreExtra"),
        "Extra particle history storage must remain false");
    require("1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Wall accuracy order must be one");
  }

  private static void disableAll(StudyFeature step, Model model) {
    for (String tag : model.component("comp1").physics().tags()) {
      step.setSolveFor("/physics/" + tag, false);
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      step.setSolveFor("/multiphysics/" + tag, false);
    }
  }

  private static String baseSolver(Model model) {
    String[] direct = model.study(STUDY).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(STUDY).getSolverSequences("All")) {
      try {
        if (STUDY.equals(model.sol(tag).study())) return tag;
      } catch (Throwable ignored) {
        // Continue until the generated sequence for this study is found.
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

  private static String runStudy(Model model, String fixedStepSeconds, String sourceReadback) throws Exception {
    model.study().create(STUDY);
    model.study(STUDY).label("M3-C3 Case-P 100 nm aggregate three-current");
    model.study(STUDY).create("time", "Transient");
    StudyFeature time = model.study(STUDY).feature("time");
    disableAll(time, model);
    time.setSolveFor("/physics/" + PHYSICS, true);
    time.set("tlist", TIME_LIST);
    time.set("usertol", true);
    time.set("rtol", "1e-8");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", BACKGROUND_STUDY);
    time.set("notstudystep", BACKGROUND_STEP);
    time.set("notsol", BACKGROUND_SOLUTION);
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");

    configurePhysics(model);
    model.study(STUDY).createAutoSequences("all");
    String base = baseSolver(model);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", fixedStepSeconds + "[s]");
    transientSolver.set("rtol", "1e-8");
    ParticleRunReadback.write(model, ".", sourceReadback,
        ParticleRunReadback.snapshot(model, PHYSICS, transientSolver, STUDY));

    long started = System.nanoTime();
    model.study(STUDY).run();
    emit("solve_pass", "step_s", fixedStepSeconds, "seconds",
        String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base_solution", base);
    return resultStore(model, base);
  }

  private static void createDataset(Model model, String solution) {
    model.result().dataset().create("partM3C3", "Particle");
    DatasetFeature dataset = model.result().dataset("partM3C3");
    dataset.set("solution", solution);
    dataset.set("posdof", new String[] {"comp1." + POSITION_R, "comp1." + POSITION_Z});
    dataset.set("geom", "geom1");
    dataset.set("pgeom", PARTICLE_GEOMETRY);
    dataset.set("pgeomspec", "fromphysics");
    dataset.set("physicsinterface", PHYSICS);
  }

  private static int particleRows(Model model) {
    String tag = "m3c3ParticleCount";
    try {
      model.result().numerical().create(tag, "Particle");
      NumericalFeature numerical = model.result().numerical(tag);
      numerical.set("data", "partM3C3");
      numerical.set("expr", PHYSICS + ".pidx");
      numerical.set("unit", "1");
      numerical.set("innerinput", "first");
      int count = 0;
      for (double[] row : numerical.getReal(false)) count += row.length;
      return count;
    } finally {
      try {
        model.result().numerical().remove(tag);
      } catch (Throwable ignored) {
        // Nothing remains when COMSOL rejected the numerical feature.
      }
    }
  }

  private static String[] configuredForce(Physics physics, String tag) {
    String[] values = physics.feature(tag).getStringArray("F");
    require(values.length >= 3, "Expected a three-component force for " + tag);
    return values;
  }

  private static String configuredChargeRate(Physics physics) {
    String[] values = physics.feature("auxq").getStringArray("R");
    require(values.length >= 1, "Missing dynamic-charge rate expression");
    return values[0];
  }

  private static String[] diagnosticStateExpressions() {
    return new String[] {
      PHYSICS + ".pidx", "t", POSITION_R, POSITION_Z, PHYSICS + ".vr", PHYSICS + ".vz",
      CHARGE, "particlestatus", PHYSICS + ".fs", PHYSICS + ".st"
    };
  }

  private static String[] diagnosticForceExpressions(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    String[] rates = collectionRates();
    String configuredRate = configuredChargeRate(physics);
    require(rates[3].equals(configuredRate), "Configured three-current total differs");
    String[] ion = configuredForce(physics, "idf");
    String[] drag = configuredForce(physics, EPSTEIN_TAG);
    String[] thermo = configuredForce(physics, THERMO_TAG);
    String[] lift = configuredForce(physics, "liftfm");
    String[] dep = configuredForce(physics, "depf");
    String electricR = PHYSICS + ".ef1.Fer";
    String electricZ = PHYSICS + ".ef1.Fez";
    String gravityR = PHYSICS + ".gf1.Fgr";
    String gravityZ = PHYSICS + ".gf1.Fgz";
    String totalR = "(" + electricR + "+(" + drag[0] + ")+" + gravityR + "+(" + ion[0]
        + ")+(" + thermo[0] + ")+(" + lift[0] + ")+(" + dep[0] + "))";
    String totalZ = "(" + electricZ + "+(" + drag[2] + ")+" + gravityZ + "+(" + ion[2]
        + ")+(" + thermo[2] + ")+(" + lift[2] + ")+(" + dep[2] + "))";
    emit("diagnostic_formula", "feature", "auxq", "R", configuredRate,
        "positive_collection_rate", rates[0], "electron_collection_rate", rates[1],
        "negative_collection_rate", rates[2]);
    emit("diagnostic_formula", "feature", "idf", "F", Arrays.toString(ion));
    emit("diagnostic_formula", "feature", EPSTEIN_TAG, "F", Arrays.toString(drag));
    emit("diagnostic_formula", "feature", THERMO_TAG, "F", Arrays.toString(thermo));
    emit("diagnostic_formula", "feature", "liftfm", "F", Arrays.toString(lift));
    emit("diagnostic_formula", "feature", "depf", "F", Arrays.toString(dep));
    return new String[] {
      PHYSICS + ".pidx", "t", configuredRate, rates[0], rates[1], rates[2],
      electricR, electricZ, ion[0], ion[2], drag[0], drag[2], thermo[0], thermo[2],
      lift[0], lift[2], dep[0], dep[2], gravityR, gravityZ, totalR, totalZ
    };
  }

  private static void exportDiagnosticTable(
      Model model,
      String tag,
      String filename,
      String[] expressions,
      String[] units,
      String[] columns) {
    require(expressions.length == units.length, filename + " expression/unit mismatch");
    require(expressions.length == columns.length, filename + " expression/column mismatch");
    model.result().export().create(tag, "Data");
    ExportFeature export = model.result().export(tag);
    export.set("data", "partM3C3");
    export.set("expr", expressions);
    export.set("unit", units);
    export.set("descr", columns);
    export.set("filename", filename);
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "all");
    export.run();
    emit("diagnostic_export_pass", "table", filename);
  }

  private static void exportDiagnostics(Model model) {
    exportDiagnosticTable(
        model,
        "m3c3DiagnosticState",
        "diagnostic_state_raw_wide.csv",
        diagnosticStateExpressions(),
        DIAGNOSTIC_STATE_UNITS,
        DIAGNOSTIC_STATE_COLUMNS);
    exportDiagnosticTable(
        model,
        "m3c3DiagnosticForce",
        "diagnostic_force_raw_wide.csv",
        diagnosticForceExpressions(model),
        DIAGNOSTIC_FORCE_UNITS,
        DIAGNOSTIC_FORCE_COLUMNS);
  }

  private static void exportHistory(Model model) {
    String[] expressions = {
      PHYSICS + ".pidx", "t", POSITION_R, POSITION_Z, PHYSICS + ".vr", PHYSICS + ".vz",
      CHARGE, "particlestatus", PHYSICS + ".fs", PHYSICS + ".st"
    };
    require(expressions.length == STATE_COLUMNS.length, "State export metadata mismatch");
    model.result().export().create("m3c3Trajectory", "Data");
    ExportFeature export = model.result().export("m3c3Trajectory");
    export.set("data", "partM3C3");
    export.set("expr", expressions);
    export.set("unit", STATE_UNITS);
    export.set("descr", STATE_COLUMNS);
    export.set("filename", "trajectory_raw_wide.csv");
    export.set("header", true);
    export.set("fullprec", true);
    export.set("includecoords", false);
    export.set("includenan", true);
    export.set("struct", "spreadsheet");
    export.set("innerinput", "all");
    export.run();
  }

  private static void run(String fixedStepSeconds) throws Exception {
    run(fixedStepSeconds, false);
  }

  private static void run(String fixedStepSeconds, boolean diagnostic) throws Exception {
    require(!diagnostic || DIAGNOSTIC_FIXED_STEP_SECONDS.equals(fixedStepSeconds),
        "Diagnostic export is restricted to the 1.25 us fine rerun");
    Model model = null;
    try {
      model = ModelUtil.loadCopy("M3C3CasePThreeCurrent", SOURCE);
      validateSource(model);
      String sourceReadback = ParticleRunReadback.snapshot(model, PHYSICS, null, null);
      createFunctions(model);
      String solution = runStudy(model, fixedStepSeconds, sourceReadback);
      createDataset(model, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == EXPECTED_TIMES,
          "Expected 121 output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 0.03) < 1e-13,
          "Unexpected final output time");
      int particles = particleRows(model);
      require(particles == EXPECTED_PARTICLES,
          "Expected 287 particles, got " + particles);
      exportHistory(model);
      ParticleRunReadback.exportRetainedTerminalBoundaries(model, "partM3C3", PHYSICS, ".");
      if (diagnostic) {
        exportDiagnostics(model);
        exportDiagnosticTable(model, "m3c3AssembledForce", "diagnostic_assembled_force_raw_wide.csv",
            new String[] {PHYSICS + ".pidx", "t", PHYSICS + ".Ftr", PHYSICS + ".Ftz"},
            new String[] {"1", "s", "N", "N"},
            new String[] {"particle_id", "time_s", "native_total_force_r_N", "native_total_force_z_N"});
      }
      emit("configuration", "case", "caseP_100nm_three_current", "step_s",
          fixedStepSeconds,
          "time_end_s", "0.03", "output_times", Integer.toString(times.length),
          "particle_rows", Integer.toString(particles), "physics", PHYSICS,
          "background_study", BACKGROUND_STUDY, "background_step", BACKGROUND_STEP,
          "background_solution", BACKGROUND_SOLUTION, "brownian_active", "false",
          "saffman_active", "false", "dynamic_charge_active", "true",
          "charge_revision", "aggregate_relative_drift_regularized_three_current_v1",
          "ion_drag_revision", "relative_flow_screened_collection_orbital_aggregate_ion_v1",
          "drag_revision", "epstein_linear_effective_gas_sensitivity_v1",
          "drag_implementation", "explicit_custom_force",
          "maximum_relative_ion_speed_m_s", "1e6", "integrator", "classical_rk4",
          "integrator_order", "4", "relative_tolerance", "1e-8",
          "field_source", "canonical_exact_connectivity_P1_sectionwise",
          "release_source", "shared_three_current_release_table", "boundary_material", "stick",
          "boundary_37", "freeze_hold", "boundary_35", "disappear_escape",
          "source_model", SOURCE, "model_saved", "false");
      emit("run_pass", "case", "caseP_100nm_three_current", "step_s", fixedStepSeconds,
          "time_end_s", "0.03", "output_times", "121", "particles", "287", "model_saved", "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
    }
  }

  /** Runs the external-V&amp;V-only 1.25 us force and charge-rate diagnostic. */
  public static void runFineDiagnostic() throws Exception {
    ModelUtil.showProgress(false);
    emit("diagnostic_launch", "case", "caseP_100nm_three_current", "source", SOURCE,
        "step_s", DIAGNOSTIC_FIXED_STEP_SECONDS, "scope", "external_vv_only");
    run(DIAGNOSTIC_FIXED_STEP_SECONDS, true);
    emit("diagnostic_run_pass", "case", "caseP_100nm_three_current", "step_s",
        DIAGNOSTIC_FIXED_STEP_SECONDS, "state", "diagnostic_state_raw_wide.csv",
        "force", "diagnostic_force_raw_wide.csv", "model_saved", "false");
  }

  public static void main(String[] args) throws Exception {
    try {
      ModelUtil.showProgress(false);
      String fixedStepSeconds = fixedStepSeconds();
      emit("launch", "case", "caseP_100nm_three_current", "source", SOURCE,
          "step_s", fixedStepSeconds);
      run(fixedStepSeconds);
    } catch (Throwable failure) {
      emit("fatal", "exception", failure.getClass().getName(), "message",
          String.valueOf(failure.getMessage()));
      failure.printStackTrace(System.out);
      if (failure instanceof Error) throw (Error) failure;
      if (failure instanceof Exception) throw (Exception) failure;
      throw new RuntimeException(failure);
    }
  }
}

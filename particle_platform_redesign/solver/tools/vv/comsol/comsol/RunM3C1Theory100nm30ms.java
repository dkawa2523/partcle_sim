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
 * Runs one deterministic 30 ms, 100 nm M3-C1 same-P1 COMSOL workflow.
 *
 * <p>The working directory contains an isolated {@code source_copy.mph}, the
 * canonical P1 sectionwise tables, release tables, and {@code run_spec.properties}.
 * The source is loaded with {@code loadCopy} and is never saved.  This class
 * deliberately owns no Case-P/Case-A native field formula: all RHS primitives
 * are read from the same canonical P1 tables used by the candidate solver.
 */
public final class RunM3C1Theory100nm30ms {
  private static final String SOURCE = "source_copy.mph";
  private static final String SPEC_CASE_NAME = "__M3C1_CASE_NAME__";
  private static final String SPEC_PHYSICS = "__M3C1_PHYSICS_TAG__";
  private static final String SPEC_BACKGROUND_STUDY = "__M3C1_BACKGROUND_STUDY__";
  private static final String SPEC_BACKGROUND_SOLUTION = "__M3C1_BACKGROUND_SOLUTION__";
  private static final String SPEC_SOURCE_DATASET = "__M3C1_SOURCE_DATASET__";
  private static final String SPEC_PARTICLE_GEOMETRY = "__M3C1_PARTICLE_GEOMETRY__";
  private static final String SPEC_POSITION_DOF_R = "__M3C1_POSITION_DOF_R__";
  private static final String SPEC_POSITION_DOF_Z = "__M3C1_POSITION_DOF_Z__";
  private static final String SPEC_CHARGE_STATE = "__M3C1_CHARGE_STATE__";
  private static final String SPEC_COMMON_CONFIG_SHA256 = "__M3C1_COMMON_CONFIG_SHA256__";
  private static final String SPEC_COMSOL_CONFIG_SHA256 = "__M3C1_COMSOL_CONFIG_SHA256__";
  private static final String TIME_LIST =
      "range(0[s],1e-5[s],5e-4[s]) range(6e-4[s],1e-4[s],5e-3[s]) "
          + "range(6e-3[s],1e-3[s],3e-2[s])";
  private static final int EXPECTED_PARTICLES = 287;
  private static final int EXPECTED_TIMES = 121;
  private static final String[] RUN_KEYS = {"coarse", "medium", "fine"};
  private static final String[] STEP_SECONDS = {
    "__M3C1_STEP_COARSE_S__", "__M3C1_STEP_MEDIUM_S__", "__M3C1_STEP_FINE_S__"
  };
  private static final String CHARGE_LIPSCHITZ_S_INV =
      "__M3C1_CHARGE_LIPSCHITZ_S_INV__";
  private static final String MAXIMUM_DT_CHARGE_LIPSCHITZ =
      "__M3C1_MAXIMUM_DT_CHARGE_LIPSCHITZ__";
  private static final String THERMO_TAG = "m3c1HeatFlux30ms";

  private static final String[] P1_NAMES = {
    "m3c1_rhog", "m3c1_mug", "m3c1_Tg", "m3c1_lambdag",
    "m3c1_ne", "m3c1_ni", "m3c1_Te", "m3c1_TiV", "m3c1_mi",
    "m3c1_lambdaD", "m3c1_lambdaIn", "m3c1_omegaPhi",
    "m3c1_ugr", "m3c1_ugz", "m3c1_Er", "m3c1_Ez",
    "m3c1_uir", "m3c1_uiz", "m3c1_gradE2r", "m3c1_gradE2z",
    "m3c1_qr", "m3c1_qz"
  };

  private static final String[] P1_UNITS = {
    "kg/m^3", "Pa*s", "K", "m", "1/m^3", "1/m^3", "V", "V", "kg",
    "m", "m", "1/s", "m/s", "m/s", "V/m", "V/m", "m/s", "m/s",
    "V^2/m^3", "V^2/m^3", "W/m^2", "W/m^2"
  };

  private static final String[] RELEASE_NAMES = {"m3c1_vr0", "m3c1_vz0", "m3c1_Z0"};
  private static final String[] RELEASE_UNITS = {"m/s", "m/s", "1"};

  private static final String[] STATE_COLUMNS = {
    "particle_id", "time_s", "r_m", "z_m", "velocity_r_m_per_s",
    "velocity_z_m_per_s", "charge_number_e", "current_status_code",
    "final_status_code", "stop_or_event_time_s", "charge_rate_e_per_s",
    "particle_mass_kg", "sampled_release_velocity_r_m_per_s",
    "sampled_release_velocity_z_m_per_s", "sampled_release_charge_number_e"
  };
  private static final String[] STATE_UNITS = {
    "1", "s", "m", "m", "m/s", "m/s", "1", "1", "1", "s", "1/s", "kg",
    "m/s", "m/s", "1"
  };

  private static final String[] FORCE_COLUMNS = {
    "particle_id", "time_s", "electric_force_r_N", "electric_force_z_N",
    "ion_drag_force_r_N", "ion_drag_force_z_N", "epstein_force_r_N",
    "epstein_force_z_N", "thermophoretic_force_r_N", "thermophoretic_force_z_N",
    "lift_force_r_N", "lift_force_z_N", "dep_force_r_N", "dep_force_z_N",
    "gravity_buoyancy_force_r_N", "gravity_buoyancy_force_z_N",
    "total_force_r_N", "total_force_z_N", "acceleration_r_m_per_s2",
    "acceleration_z_m_per_s2"
  };
  private static final String[] FORCE_UNITS = {
    "1", "s", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N",
    "N", "N", "N", "N", "N", "N", "m/s^2", "m/s^2"
  };

  private static final String[] PRIMITIVE_COLUMNS = {
    "particle_id", "time_s", "gas_density_kg_per_m3",
    "gas_dynamic_viscosity_Pa_s", "gas_temperature_K", "gas_mean_free_path_m",
    "electron_density_per_m3", "positive_ion_density_per_m3",
    "electron_thermal_energy_eV_as_V", "positive_ion_thermal_energy_eV_as_V",
    "effective_positive_ion_mass_kg", "screening_length_m",
    "ion_neutral_mean_free_path_m", "azimuthal_vorticity_per_s",
    "gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s", "electric_field_r_V_per_m",
    "electric_field_z_V_per_m", "ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s",
    "gradient_E2_r_V2_per_m3", "gradient_E2_z_V2_per_m3",
    "gas_translational_heat_flux_r_W_per_m2",
    "gas_translational_heat_flux_z_W_per_m2"
  };
  private static final String[] PRIMITIVE_UNITS = {
    "1", "s", "kg/m^3", "Pa*s", "K", "m", "1/m^3", "1/m^3", "V", "V",
    "kg", "m", "m", "1/s", "m/s", "m/s", "V/m", "V/m", "m/s", "m/s",
    "V^2/m^3", "V^2/m^3", "W/m^2", "W/m^2"
  };

  private RunM3C1Theory100nm30ms() {}

  private static String required(String value, String key) {
    if (value == null || value.trim().isEmpty()) {
      throw new IllegalStateException("Missing run specification property " + key);
    }
    return value.trim();
  }

  private static double positiveFinite(String value, String key) {
    double parsed;
    try {
      parsed = Double.parseDouble(required(value, key));
    } catch (NumberFormatException error) {
      throw new IllegalStateException("Invalid numeric run specification " + key, error);
    }
    require(!Double.isNaN(parsed) && !Double.isInfinite(parsed) && parsed > 0.0,
        "Run specification must be positive and finite: " + key);
    return parsed;
  }

  private static void validateEmbeddedSpec() {
    required(SPEC_CASE_NAME, "case_name");
    required(SPEC_PHYSICS, "physics_tag");
    required(SPEC_BACKGROUND_STUDY, "background_study");
    required(SPEC_BACKGROUND_SOLUTION, "background_solution");
    required(SPEC_SOURCE_DATASET, "source_dataset");
    required(SPEC_PARTICLE_GEOMETRY, "particle_geometry");
    required(SPEC_POSITION_DOF_R, "position_dof_r");
    required(SPEC_POSITION_DOF_Z, "position_dof_z");
    required(SPEC_CHARGE_STATE, "charge_state");
    required(SPEC_COMMON_CONFIG_SHA256, "common_config_sha256");
    required(SPEC_COMSOL_CONFIG_SHA256, "comsol_config_sha256");
    require(SPEC_CASE_NAME.equals("caseP") || SPEC_CASE_NAME.equals("caseA"),
        "Unsupported case");
    require(Arrays.equals(RUN_KEYS, new String[] {"coarse", "medium", "fine"}),
        "Run keys must be coarse, medium, fine");
    require(STEP_SECONDS.length == RUN_KEYS.length, "Fixed-step array length differs");
    double[] steps = new double[STEP_SECONDS.length];
    for (int index = 0; index < STEP_SECONDS.length; index++) {
      steps[index] = positiveFinite(STEP_SECONDS[index], RUN_KEYS[index] + "_step_s");
      double count = 0.03 / steps[index];
      require(Math.abs(count - Math.rint(count)) <= 8.0 * Math.ulp(count),
          RUN_KEYS[index] + " step does not divide the 30 ms interval");
    }
    require(steps[0] == 2.0 * steps[1] && steps[1] == 2.0 * steps[2],
        "Fixed RK4 steps must be an exact h, h/2, h/4 sequence");
    double lipschitz = positiveFinite(CHARGE_LIPSCHITZ_S_INV, "charge_lipschitz_s_inv");
    double maximum = positiveFinite(
        MAXIMUM_DT_CHARGE_LIPSCHITZ, "maximum_dt_charge_lipschitz");
    for (int index = 0; index < steps.length; index++) {
      require(steps[index] * lipschitz <= maximum,
          RUN_KEYS[index] + " step violates the charge Lipschitz limit");
    }
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C1_30MS|").append(type);
    for (int index = 0; index + 1 < values.length; index += 2) {
      line.append('|').append(values[index]).append('=').append(values[index + 1]);
    }
    String text = line.toString();
    System.out.println(text);
    ModelUtil.serverLog(text);
  }

  private static String stepExpression(int index) {
    return STEP_SECONDS[index] + "[s]";
  }

  private static String stepSeconds(int index) {
    return STEP_SECONDS[index];
  }

  private static String stepDirectory(int index) {
    return RUN_KEYS[index];
  }

  private static void requireFeature(Physics physics, String tag) {
    require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
  }

  private static void requireEntities(PhysicsFeature feature, int... expected) {
    int[] actual = feature.selection().entities();
    require(Arrays.equals(actual, expected),
        "Unexpected entity selection for " + feature.tag() + ": " + Arrays.toString(actual));
  }

  private static void validateSource(Model model) {
    require(has(model.component("comp1").physics().tags(), SPEC_PHYSICS),
        "Missing source physics " + SPEC_PHYSICS);
    require(has(model.study().tags(), SPEC_BACKGROUND_STUDY),
        "Missing source background study " + SPEC_BACKGROUND_STUDY);
    require(has(model.sol().tags(), SPEC_BACKGROUND_SOLUTION),
        "Missing source background solution " + SPEC_BACKGROUND_SOLUTION);
    require(!model.sol(SPEC_BACKGROUND_SOLUTION).isEmpty(), "Source background solution is empty");
    require(has(model.result().dataset().tags(), SPEC_SOURCE_DATASET),
        "Missing source particle dataset " + SPEC_SOURCE_DATASET);
    DatasetFeature dataset = model.result().dataset(SPEC_SOURCE_DATASET);
    require(SPEC_PARTICLE_GEOMETRY.equals(dataset.getString("pgeom")),
        "Particle geometry tag differs");
    require(SPEC_PHYSICS.equals(dataset.getString("physicsinterface")),
        "Particle dataset physics differs");
    require(Arrays.equals(
        new String[] {"comp1." + SPEC_POSITION_DOF_R, "comp1." + SPEC_POSITION_DOF_Z},
        dataset.getStringArray("posdof")), "Particle position DOFs differ");

    Physics physics = model.component("comp1").physics(SPEC_PHYSICS);
    for (String tag : new String[] {
        "relg1", "pp1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm",
        "depf", "gf1", "bf1", "lf1", "wall1", "outin", "outpump", "axi1"
    }) requireFeature(physics, tag);
    requireEntities(physics.feature("wall1"), 6, 8, 28, 29, 32, 33, 34, 36, 38, 40, 41,
        45, 46, 47);
    requireEntities(physics.feature("outin"), 37);
    requireEntities(physics.feature("outpump"), 35);
    requireEntities(physics.feature("axi1"), 5);
  }

  private static void createFunctions(Model model) {
    require(P1_NAMES.length == P1_UNITS.length, "P1 function metadata mismatch");
    for (int index = 0; index < P1_NAMES.length; index++) {
      String tag = "m3c1LongP1F" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", P1_NAMES[index] + "_sectionwise.txt");
      function.set("struct", "sectionwise");
      function.set("funcs", new String[][] {{P1_NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", P1_UNITS[index]);
      function.importData();
    }
    for (int index = 0; index < RELEASE_NAMES.length; index++) {
      String tag = "m3c1LongReleaseF" + index;
      model.func().create(tag, "Interpolation");
      FunctionFeature function = model.func(tag);
      function.set("source", "file");
      function.set("filename", RELEASE_NAMES[index] + ".txt");
      function.set("struct", "spreadsheet");
      function.set("funcs", new String[][] {{RELEASE_NAMES[index], "1"}});
      function.set("interp", "linear");
      function.set("extrap", "const");
      function.set("argunit", new String[] {"m", "m"});
      function.set("fununit", RELEASE_UNITS[index]);
      function.importData();
    }
  }

  private static String velocitySquared() {
    return "((m3c1_uir(r,z)-" + SPEC_PHYSICS + ".vr)^2"
        + "+(m3c1_uiz(r,z)-" + SPEC_PHYSICS + ".vz)^2"
        + "+8*e_const*m3c1_TiV(r,z)/(pi*m3c1_mi(r,z))+u_eps^2)";
  }

  private static String particlePotential() {
    return "(" + SPEC_CHARGE_STATE + "*e_const/(4*pi*epsilon0_const*(d0/2)"
        + "*(1+(d0/2)/m3c1_lambdaD(r,z))))";
  }

  private static String chargeRate() {
    String speed2 = velocitySquared();
    String potential = particlePotential();
    String ionEnergy = "max(m3c1_mi(r,z)*" + speed2 + "/(2*e_const),Ti_floor)";
    String ionCorrection = "if(" + potential + "<=0[V],1-" + potential + "/(" + ionEnergy
        + "),exp(max(-50,min(50,-" + potential + "/(" + ionEnergy + ")))))";
    String electronCorrection = "if(" + potential + "<=0[V],exp(max(-50,min(50,"
        + potential + "/m3c1_Te(r,z)))),1+" + potential + "/m3c1_Te(r,z))";
    String ion = "pi*(d0/2)^2*m3c1_ni(r,z)*sqrt(" + speed2 + ")*(" + ionCorrection + ")";
    String electron = "pi*(d0/2)^2*m3c1_ne(r,z)"
        + "*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const))*(" + electronCorrection + ")";
    return "(" + ion + "-(" + electron + "))";
  }

  private static String[] ionDrag() {
    String speed2 = velocitySquared();
    String potential = particlePotential();
    String screen = "max(d0/2,min(m3c1_lambdaD(r,z),m3c1_lambdaIn(r,z)))";
    String capture = "min((" + screen + ")^2,(d0/2)^2*max(0,1-2*e_const*"
        + potential + "/(m3c1_mi(r,z)*" + speed2 + ")))";
    String b90 = "sqrt(" + SPEC_CHARGE_STATE + "^2+1e-20)*e_const^2/"
        + "(4*pi*epsilon0_const*m3c1_mi(r,z)*" + speed2 + ")";
    String coulombLog = "max(0,0.5*log(((" + screen + ")^2+(" + b90 + ")^2)/("
        + capture + "+(" + b90 + ")^2)))";
    String scalar = "(pi*(" + capture + ")+4*pi*(" + b90 + ")^2*(" + coulombLog + "))"
        + "*m3c1_ni(r,z)*m3c1_mi(r,z)*sqrt(" + speed2 + ")";
    return new String[] {
      scalar + "*(m3c1_uir(r,z)-" + SPEC_PHYSICS + ".vr)",
      "0[N]",
      scalar + "*(m3c1_uiz(r,z)-" + SPEC_PHYSICS + ".vz)"
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

  private static String[] lift() {
    String common = "pi*m3c1_rhog(r,z)*m3c1_lambdag(r,z)*(d0/2)^2";
    return new String[] {
      common + "*(m3c1_ugz(r,z)-" + SPEC_PHYSICS + ".vz)*m3c1_omegaPhi(r,z)",
      "0[N]",
      "-" + common + "*(m3c1_ugr(r,z)-" + SPEC_PHYSICS + ".vr)*m3c1_omegaPhi(r,z)"
    };
  }

  private static String[] dep() {
    String factor = "2*pi*epsilon0_const*(d0/2)^3*0.5161290322580645";
    return new String[] {
      factor + "*m3c1_gradE2r(r,z)", "0[N]", factor + "*m3c1_gradE2z(r,z)"
    };
  }

  private static void configureForce(
      PhysicsFeature feature, String label, String[] force, String study) {
    feature.label(label);
    feature.selection().set(3);
    feature.set("SpecifyForce", "Directly");
    feature.set("F", force);
    feature.set("ParticlesToAffect", "All");
    feature.set("AffectedParticleProperties", "pp1");
    feature.set("StudyStep", study + "/time");
  }

  private static void configurePhysics(Model model, String study) {
    Physics physics = model.component("comp1").physics(SPEC_PHYSICS);
    physics.feature("bf1").active(false);
    physics.feature("lf1").active(false);
    physics.feature("thpf1").active(false);
    for (String tag : new String[] {"auxq", "idf", "ef1", "df1", "liftfm", "depf", "gf1"}) {
      physics.feature(tag).active(true);
    }

    model.param().set("d0", "100[nm]");
    model.param().set("sigmaR_p", "0.9");
    physics.feature("relg1").set(
        "v0", new String[] {"m3c1_vr0(r,z)", "0[m/s]", "m3c1_vz0(r,z)"});
    physics.feature("relg1").set("aux0_auxq", "m3c1_Z0(r,z)");
    physics.feature("pp1").set("ChargeSpecification", "UserDefined");
    physics.feature("pp1").set("Z", SPEC_CHARGE_STATE);

    physics.feature("auxq").set("R", new String[] {chargeRate()});
    physics.feature("auxq").set("StudyStep", study + "/time");
    configureForce(physics.feature("idf"), "M3-C1 exact-P1 theory ion drag", ionDrag(), study);
    physics.feature("ef1").set(
        "E", new String[] {"m3c1_Er(r,z)", "0[V/m]", "m3c1_Ez(r,z)"});
    physics.feature("ef1").set("StudyStep", study + "/time");

    PhysicsFeature drag = physics.feature("df1");
    drag.set("u", new String[] {"m3c1_ugr(r,z)", "0[m/s]", "m3c1_ugz(r,z)"});
    drag.set("rho", "m3c1_rhog(r,z)");
    drag.set("mu", "m3c1_mug(r,z)");
    drag.set("minput_temperature", "m3c1_Tg(r,z)");
    String pressure =
        "m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]";
    drag.set("pA", pressure);
    drag.set("minput_pressure", pressure);
    drag.set("S", "1.0");
    drag.set("sigmaR", "sigmaR_p");
    drag.set("StudyStep", study + "/time");

    configureForce(physics.feature("liftfm"), "M3-C1 exact-P1 rarefied lift", lift(), study);
    configureForce(physics.feature("depf"), "M3-C1 exact-P1 DEP", dep(), study);
    require(!has(physics.feature().tags(), THERMO_TAG), "Thermophoresis force tag already exists");
    physics.create(THERMO_TAG, "Force", 2);
    configureForce(
        physics.feature(THERMO_TAG), "M3-C1 exact-P1 Waldmann heat-flux force",
        thermophoresis(), study);

    physics.feature("gf1").set("rho", "m3c1_rhog(r,z)");
    physics.feature("gf1").set("minput_temperature", "m3c1_Tg(r,z)");
    physics.feature("gf1").set("StudyStep", study + "/time");

    physics.feature("wall1").set("WallCondition", "Stick");
    physics.feature("outin").set("WallCondition", "Freeze");
    physics.feature("outpump").set("WallCondition", "Disappear");
    for (String tag : new String[] {
        "wall1", "outin", "outpump", "axi1", "relg1", "pp1"
    }) physics.feature(tag).set("StudyStep", study + "/time");

    require(!physics.feature("bf1").isActive(), "Brownian force remained active");
    require(!physics.feature("lf1").isActive(), "Saffman lift remained active");
    require(!physics.feature("thpf1").isActive(), "Native thermophoresis remained active");
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "liftfm", "depf", "gf1", THERMO_TAG,
        "relg1", "pp1", "wall1", "outin", "outpump", "axi1"
    }) require(physics.feature(tag).isActive(), "Required feature is inactive: " + tag);
    require("Stick".equals(physics.feature("wall1").getString("WallCondition")),
        "Material wall is not Stick");
    require("Freeze".equals(physics.feature("outin").getString("WallCondition")),
        "Gas inlet is not Freeze/hold");
    require("Disappear".equals(physics.feature("outpump").getString("WallCondition")),
        "Pump outlet is not Disappear/escape");
    require(physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage is disabled");
    require(!physics.prop("StoreExtra").getBoolean("StoreExtra"),
        "Extra particle history storage must remain disabled");
    require("1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Wall event localization must use first-order accuracy");
  }

  private static void allOff(StudyFeature step, Model model) {
    for (String tag : model.component("comp1").physics().tags()) {
      try {
        step.setSolveFor("/physics/" + tag, false);
      } catch (Throwable ignored) {
        // Some source interfaces do not expose solve-for control in this study type.
      }
    }
    for (String tag : model.component("comp1").multiphysics().tags()) {
      try {
        step.setSolveFor("/multiphysics/" + tag, false);
      } catch (Throwable ignored) {
        // Couplings not owned by the particle study remain disabled.
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
        // Continue until the generated sequence for this study is found.
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

  private static String runStudy(Model model, int stepIndex) {
    String study = "stdM3C1Long";
    model.param().set("M3C1_long_dt", stepExpression(stepIndex));
    model.study().create(study);
    model.study(study).label("M3-C1 exact-P1 " + SPEC_CASE_NAME + " 100 nm 30 ms");
    model.study(study).create("param", "Parametric");
    StudyFeature parameter = model.study(study).feature("param");
    parameter.set("pname", new String[] {"d0"});
    parameter.set("plistarr", new String[] {"100"});
    parameter.set("punit", new String[] {"nm"});
    parameter.set("sweeptype", "sparse");

    model.study(study).create("time", "Transient");
    StudyFeature time = model.study(study).feature("time");
    allOff(time, model);
    time.setSolveFor("/physics/" + SPEC_PHYSICS, true);
    time.set("tlist", TIME_LIST);
    time.set("usertol", true);
    time.set("rtol", "1e-8");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", SPEC_BACKGROUND_STUDY);
    time.set("notsol", SPEC_BACKGROUND_SOLUTION);
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");

    configurePhysics(model, study);
    model.study(study).createAutoSequences("all");
    String base = baseSolver(model, study);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3C1_long_dt");
    transientSolver.set("rtol", "1e-8");

    long started = System.nanoTime();
    model.study(study).run();
    emit(
        "solve_pass", "case", SPEC_CASE_NAME, "run_key", RUN_KEYS[stepIndex],
        "step_s", stepSeconds(stepIndex),
        "seconds", String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
        "base_solution", base);
    return resultStore(model, study, base);
  }

  private static void createDataset(Model model, String tag, String solution) {
    model.result().dataset().create(tag, "Particle");
    DatasetFeature dataset = model.result().dataset(tag);
    dataset.set("solution", solution);
    dataset.set("posdof", new String[] {
        "comp1." + SPEC_POSITION_DOF_R, "comp1." + SPEC_POSITION_DOF_Z
    });
    dataset.set("geom", "geom1");
    dataset.set("pgeom", SPEC_PARTICLE_GEOMETRY);
    dataset.set("pgeomspec", "fromphysics");
    dataset.set("physicsinterface", SPEC_PHYSICS);
  }

  private static String[] force(Physics physics, String tag) {
    String[] values = physics.feature(tag).getStringArray("F");
    require(values.length >= 3, "Expected a three-component force for " + tag);
    return values;
  }

  private static String[] stateExpressions(Model model) {
    Physics physics = model.component("comp1").physics(SPEC_PHYSICS);
    String[] rates = physics.feature("auxq").getStringArray("R");
    require(rates.length >= 1, "Missing dynamic-charge rate");
    return new String[] {
      SPEC_PHYSICS + ".pidx", "t", SPEC_POSITION_DOF_R, SPEC_POSITION_DOF_Z,
      SPEC_PHYSICS + ".vr", SPEC_PHYSICS + ".vz", SPEC_CHARGE_STATE,
      "particlestatus", SPEC_PHYSICS + ".fs", SPEC_PHYSICS + ".st", rates[0],
      "rho_p*pi*d0^3/6", "m3c1_vr0(" + SPEC_POSITION_DOF_R + ","
          + SPEC_POSITION_DOF_Z + ")", "m3c1_vz0(" + SPEC_POSITION_DOF_R + ","
          + SPEC_POSITION_DOF_Z + ")", "m3c1_Z0(" + SPEC_POSITION_DOF_R + ","
          + SPEC_POSITION_DOF_Z + ")"
    };
  }

  private static String[] forceExpressions(Model model) {
    Physics physics = model.component("comp1").physics(SPEC_PHYSICS);
    String[] ion = force(physics, "idf");
    String[] thermo = force(physics, THERMO_TAG);
    String[] rarefiedLift = force(physics, "liftfm");
    String[] dielectrophoresis = force(physics, "depf");
    String electricR = SPEC_PHYSICS + ".ef1.Fer";
    String electricZ = SPEC_PHYSICS + ".ef1.Fez";
    String dragR = SPEC_PHYSICS + ".df1.FDr";
    String dragZ = SPEC_PHYSICS + ".df1.FDz";
    String gravityR = SPEC_PHYSICS + ".gf1.Fgr";
    String gravityZ = SPEC_PHYSICS + ".gf1.Fgz";
    String totalR = "(" + electricR + "+(" + ion[0] + ")+" + dragR + "+(" + thermo[0]
        + ")+(" + rarefiedLift[0] + ")+(" + dielectrophoresis[0] + ")+" + gravityR + ")";
    String totalZ = "(" + electricZ + "+(" + ion[2] + ")+" + dragZ + "+(" + thermo[2]
        + ")+(" + rarefiedLift[2] + ")+(" + dielectrophoresis[2] + ")+" + gravityZ + ")";
    String mass = "(rho_p*pi*d0^3/6)";
    return new String[] {
      SPEC_PHYSICS + ".pidx", "t", electricR, electricZ, ion[0], ion[2], dragR, dragZ,
      thermo[0], thermo[2], rarefiedLift[0], rarefiedLift[2], dielectrophoresis[0],
      dielectrophoresis[2], gravityR, gravityZ, totalR, totalZ, totalR + "/" + mass,
      totalZ + "/" + mass
    };
  }

  private static String[] primitiveExpressions() {
    String[] values = new String[P1_NAMES.length + 2];
    values[0] = SPEC_PHYSICS + ".pidx";
    values[1] = "t";
    for (int index = 0; index < P1_NAMES.length; index++) {
      values[index + 2] = P1_NAMES[index] + "(" + SPEC_POSITION_DOF_R + ","
          + SPEC_POSITION_DOF_Z + ")";
    }
    return values;
  }

  private static void exportTable(
      Model model, String dataset, String directory, String tag, String filename,
      String[] expressions, String[] units, String[] columns) {
    require(expressions.length == units.length, filename + " expression/unit mismatch");
    require(expressions.length == columns.length, filename + " expression/column mismatch");
    model.result().export().create(tag, "Data");
    ExportFeature export = model.result().export(tag);
    export.set("data", dataset);
    export.set("expr", expressions);
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
      // A nonparametric result store has no outer solution selector.
    }
    export.run();
    emit("export_pass", "case", dataset, "table", filename);
  }

  private static int particleRows(Model model, String dataset) {
    String tag = "m3c1LongParticleCount";
    try {
      model.result().numerical().create(tag, "Particle");
      NumericalFeature numerical = model.result().numerical(tag);
      numerical.set("data", dataset);
      numerical.set("expr", SPEC_PHYSICS + ".pidx");
      numerical.set("unit", "1");
      numerical.set("innerinput", "first");
      int count = 0;
      for (double[] row : numerical.getReal(false)) count += row.length;
      return count;
    } finally {
      try {
        model.result().numerical().remove(tag);
      } catch (Throwable ignored) {
        // Nothing remains to remove when COMSOL rejects the numerical feature.
      }
    }
  }

  private static void runOne(int stepIndex) throws Exception {
    Model model = null;
    try {
      model = ModelUtil.loadCopy("M3C1Long" + SPEC_CASE_NAME + stepIndex, SOURCE);
      validateSource(model);
      createFunctions(model);
      String solution = runStudy(model, stepIndex);
      String dataset = "partM3C1Long";
      createDataset(model, dataset, solution);
      double[] times = model.sol(solution).getPVals();
      require(times.length == EXPECTED_TIMES,
          "Expected 121 output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(Math.abs(times[times.length - 1] - 0.03) < 1e-14,
          "Unexpected final output time");
      int particles = particleRows(model, dataset);
      require(particles == EXPECTED_PARTICLES,
          "Expected 287 released particles, got " + particles);

      String directory = stepDirectory(stepIndex);
      exportTable(model, dataset, directory, "m3c1LongState", "state_raw_wide.csv",
          stateExpressions(model), STATE_UNITS, STATE_COLUMNS);
      exportTable(model, dataset, directory, "m3c1LongForce", "force_raw_wide.csv",
          forceExpressions(model), FORCE_UNITS, FORCE_COLUMNS);
      exportTable(model, dataset, directory, "m3c1LongPrimitive", "primitive_raw_wide.csv",
          primitiveExpressions(), PRIMITIVE_UNITS, PRIMITIVE_COLUMNS);

      emit(
          "configuration", "case", SPEC_CASE_NAME, "run_key", RUN_KEYS[stepIndex],
          "step_s", stepSeconds(stepIndex),
          "charge_lipschitz_s_inv", CHARGE_LIPSCHITZ_S_INV,
          "dt_charge_lipschitz", Double.toString(
              positiveFinite(STEP_SECONDS[stepIndex], RUN_KEYS[stepIndex] + "_step_s")
                  * positiveFinite(CHARGE_LIPSCHITZ_S_INV, "charge_lipschitz_s_inv")),
          "maximum_dt_charge_lipschitz", MAXIMUM_DT_CHARGE_LIPSCHITZ,
          "time_start_s", "0", "time_end_s", "0.03", "output_times", "121",
          "particle_rows", Integer.toString(particles), "physics", SPEC_PHYSICS,
          "background_study", SPEC_BACKGROUND_STUDY, "background_solution",
          SPEC_BACKGROUND_SOLUTION, "particle_geometry", SPEC_PARTICLE_GEOMETRY,
          "position_dofs", SPEC_POSITION_DOF_R + "," + SPEC_POSITION_DOF_Z,
          "charge_state", SPEC_CHARGE_STATE,
          "brownian_active", "false", "saffman_active", "false",
          "dynamic_charge_active", "true", "integrator", "classical_rk4",
          "integrator_order", "4", "relative_tolerance", "1e-8",
          "field_source", "canonical_exact_connectivity_P1_sectionwise",
          "boundary_material", "stick", "boundary_37", "freeze_hold",
          "boundary_35", "disappear_escape", "escape_hit_position",
          "NOT_DIRECTLY_OBSERVED_WHEN_NAN", "common_config_sha256",
          SPEC_COMMON_CONFIG_SHA256, "comsol_config_sha256", SPEC_COMSOL_CONFIG_SHA256,
          "source_model", SOURCE, "model_saved", "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  public static void main(String[] args) throws Exception {
    try {
      ModelUtil.showProgress(false);
      validateEmbeddedSpec();
      emit("launch", "case", SPEC_CASE_NAME, "source", SOURCE);
      for (int stepIndex = 0; stepIndex < RUN_KEYS.length; stepIndex++) runOne(stepIndex);
      emit(
          "run_pass", "case", SPEC_CASE_NAME, "run_keys",
          String.join(",", RUN_KEYS), "steps_s", String.join(",", STEP_SECONDS),
          "charge_lipschitz_s_inv", CHARGE_LIPSCHITZ_S_INV,
          "maximum_dt_charge_lipschitz", MAXIMUM_DT_CHARGE_LIPSCHITZ,
          "time_end_s", "0.03",
          "output_times", "121", "particles", "287", "model_saved", "false");
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

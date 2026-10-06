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
 * Runs the M3-C1 Case-A 100 nm full-physics common-P1 diagnostic.
 *
 * <p>The source MPH supplies geometry, release sites, wall semantics, and the
 * particle-interface topology. Every spatial primitive consumed by the
 * particle RHS is rebound to a sectionwise P1 function prepared from the
 * candidate HDF5. Brownian and Saffman lift remain disabled. The source model
 * is loaded from an isolated copy and is never saved. Campaign runs consume
 * only the 22 named P1 primitives plus the size-specific release state;
 * producer-derived particle-background columns are never bound or trusted.
 */
public final class RunM3C1CaseA100CommonP1 {
  private static final String SOURCE = "source_copy.mph";
  private static final String PHYSICS = "fptas";
  private static final String BACKGROUND_STUDY = "stdASf";
  private static final String BACKGROUND_SOLUTION = "sol26";
  private static final String PRE_EVENT_PROFILE = "pre_event";
  private static final String MATERIAL_EVENT_PROFILE = "material_event";
  private static final String PRE_EVENT_TLIST = "range(0[s],1e-5[s],4.5e-4[s])";
  private static final String MATERIAL_EVENT_TLIST =
      "range(0[s],1e-5[s],4.5e-4[s]) 4.58e-4[s] 4.5875e-4[s]";
  private static final String RELATIVE_FLOW_REVISION =
      "relative_flow_screened_collection_orbital_aggregate_ion_v1";
  private static final String IMAGE_REVISION =
      "electric_field_directed_image_orbital_sensitivity_v1";
  private static final String RELATIVE_FLOW_CONTRIBUTION =
      "relative_flow_screened_collection_orbital_ion_drag";
  private static final String IMAGE_CONTRIBUTION =
      "electric_field_directed_image_orbital_ion_drag";
  private static final int SPEC_CASE_ID = 0;
  private static final int SPEC_DIAMETER_NM = 1;
  private static final int SPEC_ION_DRAG_REVISION = 2;
  private static final int SPEC_CONTRIBUTION = 3;
  private static final int[] PRE_EVENT_STEP_CODES = {625, 3125, 15625};
  private static final int[] MATERIAL_EVENT_STEP_CODES = {15625};

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

  private static final String THERMO_R =
      "(32/15)*(d0/2)^2*m3c1_qr(r,z)/"
          + "sqrt(8*k_B_const*m3c1_Tg(r,z)/(pi*1.2753471408396638e-25[kg]))";
  private static final String THERMO_Z =
      "(32/15)*(d0/2)^2*m3c1_qz(r,z)/"
          + "sqrt(8*k_B_const*m3c1_Tg(r,z)/(pi*1.2753471408396638e-25[kg]))";
  private static final String LIFT_R =
      "pi*m3c1_rhog(r,z)*m3c1_lambdag(r,z)*(d0/2)^2*"
          + "(m3c1_ugz(r,z)-fptas.vz)*m3c1_omegaPhi(r,z)";
  private static final String LIFT_Z =
      "-pi*m3c1_rhog(r,z)*m3c1_lambdag(r,z)*(d0/2)^2*"
          + "(m3c1_ugr(r,z)-fptas.vr)*m3c1_omegaPhi(r,z)";
  private static final String DEP_FACTOR =
      "2*pi*epsilon0_const*1*(d0/2)^3*0.5161290322580645";
  private static final String DEP_R = DEP_FACTOR + "*m3c1_gradE2r(r,z)";
  private static final String DEP_Z = DEP_FACTOR + "*m3c1_gradE2z(r,z)";
  private static final String IMAGE_EPSILON = "8.8541878128e-12[F/m]";
  private static final String IMAGE_ION_SPEED =
      "sqrt(m3c1_uir(r,z)^2+m3c1_uiz(r,z)^2)";
  private static final String IMAGE_SPEED_SQUARE =
      "(" + IMAGE_ION_SPEED + "^2+8*e_const*m3c1_TiV(r,z)/(pi*m3c1_mi(r,z))"
          + "+(1[m/s])^2)";
  private static final String IMAGE_CAPACITANCE_SCREENING =
      "max(d0/2,m3c1_lambdaD(r,z))";
  private static final String IMAGE_SURFACE_POTENTIAL =
      "(ZAS*e_const/(4*pi*" + IMAGE_EPSILON + "*(d0/2)*(1+(d0/2)/"
          + IMAGE_CAPACITANCE_SCREENING + ")))";
  private static final String IMAGE_COLLECTION_CROSS_SECTION =
      "(pi*(d0/2)^2*max(0,1-" + IMAGE_SURFACE_POTENTIAL + "/m3c1_TiV(r,z)))";
  private static final String IMAGE_IMPACT =
      "(e_const^2*ZAS/(2*pi*" + IMAGE_EPSILON + "*m3c1_mi(r,z)*"
          + IMAGE_SPEED_SQUARE + "))";
  private static final String IMAGE_SCREENING =
      "sqrt(" + IMAGE_EPSILON + "*m3c1_Te(r,z)/(e_const*m3c1_ni(r,z)))";
  private static final String IMAGE_ORBITAL_CROSS_SECTION =
      "(pi*" + IMAGE_IMPACT + "^2*log(max(1+1e-12," + IMAGE_SCREENING + "/(d0/2))))";
  private static final String IMAGE_FORCE_MAGNITUDE =
      "(m3c1_mi(r,z)*m3c1_ni(r,z)*sqrt(" + IMAGE_SPEED_SQUARE + ")*"
          + IMAGE_ION_SPEED + "*(" + IMAGE_COLLECTION_CROSS_SECTION + "+"
          + IMAGE_ORBITAL_CROSS_SECTION + "))";
  private static final String IMAGE_ELECTRIC_NORM =
      "sqrt(m3c1_Er(r,z)^2+m3c1_Ez(r,z)^2+(1[V/m])^2)";
  private static final String IMAGE_FORCE_R =
      IMAGE_FORCE_MAGNITUDE + "*m3c1_Er(r,z)/" + IMAGE_ELECTRIC_NORM;
  private static final String IMAGE_FORCE_Z =
      IMAGE_FORCE_MAGNITUDE + "*m3c1_Ez(r,z)/" + IMAGE_ELECTRIC_NORM;

  private static String[] legacy100Relative() {
    return new String[] {
      "caseA_100nm", "100", RELATIVE_FLOW_REVISION, "relative_flow_ion_drag"
    };
  }

  private static String[] campaignSpec(
      String caseId, int diameterNm, String revision, String contribution) {
    require(caseId != null && !caseId.isEmpty(), "case_id must not be empty");
    require(revision != null && !revision.isEmpty(), "ion_drag_revision must not be empty");
    require(
        contribution != null && !contribution.isEmpty(),
        "deterministic_contribution_name must not be empty");
    boolean relative =
        ("caseA_10nm_relative_flow".equals(caseId) && diameterNm == 10)
            || ("caseA_30nm_relative_flow".equals(caseId) && diameterNm == 30);
    boolean image = "caseA_100nm_image".equals(caseId) && diameterNm == 100;
    require(relative || image, "Unsupported common-P1 campaign case tuple");
    require(
        (relative && RELATIVE_FLOW_REVISION.equals(revision))
            || (image && IMAGE_REVISION.equals(revision)),
        "ion_drag_revision does not match the supported case tuple");
    require(
        (relative && RELATIVE_FLOW_CONTRIBUTION.equals(contribution))
            || (image && IMAGE_CONTRIBUTION.equals(contribution)),
        "deterministic_contribution_name does not match the supported case tuple");
    return new String[] {caseId, Integer.toString(diameterNm), revision, contribution};
  }

  private static final String[] STATE_COLUMNS = {
    "particle_id", "time_s", "r_m", "z_m", "velocity_r_m_per_s",
    "velocity_z_m_per_s", "charge_number_e", "current_status_code",
    "final_status_code", "stop_or_event_time_s", "charge_rate_e_per_s",
    "particle_mass_kg", "sampled_release_function_velocity_r_m_per_s",
    "sampled_release_function_velocity_z_m_per_s", "sampled_release_function_charge_number_e"
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

  private static boolean has(String[] values, String target) {
    return Arrays.asList(values).contains(target);
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new IllegalStateException(message);
  }

  private static boolean isMaterialEventProfile(String profile) {
    return MATERIAL_EVENT_PROFILE.equals(profile);
  }

  private static String timeList(String profile) {
    return isMaterialEventProfile(profile) ? MATERIAL_EVENT_TLIST : PRE_EVENT_TLIST;
  }

  private static int[] stepCodes(String profile) {
    return isMaterialEventProfile(profile) ? MATERIAL_EVENT_STEP_CODES : PRE_EVENT_STEP_CODES;
  }

  private static int expectedOutputTimes(String profile) {
    return isMaterialEventProfile(profile) ? 48 : 46;
  }

  private static double expectedEndTime(String profile) {
    return isMaterialEventProfile(profile) ? 4.5875e-4 : 4.5e-4;
  }

  private static void emit(String type, String... values) {
    StringBuilder line = new StringBuilder("M3C1_COMMON_P1|").append(type);
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
    return "0.625[us]";
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

  private static void createSectionwiseFunctions(Model model) {
    require(P1_NAMES.length == P1_UNITS.length, "P1 function metadata mismatch");
    for (int index = 0; index < P1_NAMES.length; index++) {
      String tag = "m3c1P1F" + index;
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
  }

  private static void createReleaseFunctions(Model model) {
    require(RELEASE_NAMES.length == RELEASE_UNITS.length, "Release function metadata mismatch");
    for (int index = 0; index < RELEASE_NAMES.length; index++) {
      String tag = "m3c1ReleaseF" + index;
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

  private static void requireFeature(Physics physics, String tag) {
    require(has(physics.feature().tags(), tag), "Missing physics feature " + tag);
  }

  private static void bindP1Variables(Model model) {
    String[][] bindings = {
      {"AS_rhog", "m3c1_rhog(r,z)"}, {"AS_mug", "m3c1_mug(r,z)"},
      {"AS_Tg", "m3c1_Tg(r,z)"}, {"AS_lambdag", "m3c1_lambdag(r,z)"},
      {"AS_ne", "m3c1_ne(r,z)"}, {"AS_ni", "m3c1_ni(r,z)"},
      {"AS_TiV", "m3c1_TiV(r,z)"}, {"AS_lambdaD", "m3c1_lambdaD(r,z)"},
      {"AS_lambda_in", "m3c1_lambdaIn(r,z)"}, {"AS_ugr", "m3c1_ugr(r,z)"},
      {"AS_ugz", "m3c1_ugz(r,z)"}, {"AS_Er", "m3c1_Er(r,z)"},
      {"AS_Ez", "m3c1_Ez(r,z)"}, {"AS_uir", "m3c1_uir(r,z)"},
      {"AS_uiz", "m3c1_uiz(r,z)"},
      {"AS_Ge0", "pi*(d0/2)^2*m3c1_ne(r,z)*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const))"},
      {"AS_phi1", "e_const/(4*pi*epsilon0_const*(d0/2)*(1+(d0/2)/m3c1_lambdaD(r,z)))"},
      {"AS_pabs", "m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]"}
    };
    for (String[] binding : bindings) {
      model.component("comp1").variable("varAS").set(binding[0], binding[1]);
    }
  }

  private static String p1ProducerExpression(String source) {
    String[][] replacements = {
      {"root.comp1.AS_lambda_in", "m3c1_lambdaIn(r,z)"},
      {"root.comp1.AS_lambdaD", "m3c1_lambdaD(r,z)"},
      {"root.comp1.AS_uir", "m3c1_uir(r,z)"},
      {"root.comp1.AS_uiz", "m3c1_uiz(r,z)"},
      {"root.comp1.AS_TiV", "m3c1_TiV(r,z)"},
      {"root.comp1.AS_ni", "m3c1_ni(r,z)"},
      {"root.comp1.AS_Ge0",
          "(pi*(d0/2)^2*m3c1_ne(r,z)*sqrt(8*e_const*m3c1_Te(r,z)/(pi*me_const)))"},
      {"root.comp1.AS_phi1",
          "(e_const/(4*pi*epsilon0_const*(d0/2)*(1+(d0/2)/m3c1_lambdaD(r,z))))"},
      {"root.comp1.AS_mi", "m3c1_mi(r,z)"},
      {"root.comp1.AS_Te", "m3c1_Te(r,z)"},
      {"AS_mi", "m3c1_mi(r,z)"},
      {"AS_Te", "m3c1_Te(r,z)"}
    };
    String result = source;
    for (String[] replacement : replacements) {
      result = result.replace(replacement[0], replacement[1]);
    }
    for (String forbidden : new String[] {
        "root.comp1.AS_lambda_in", "root.comp1.AS_lambdaD", "root.comp1.AS_uir",
        "root.comp1.AS_uiz", "root.comp1.AS_TiV", "root.comp1.AS_ni",
        "root.comp1.AS_Ge0", "root.comp1.AS_phi1", "root.comp1.AS_mi",
        "root.comp1.AS_Te"
    }) require(!result.contains(forbidden), "Native producer field remained in expression: " + forbidden);
    return result;
  }

  private static void bindProducerFormulas(Physics physics, String[] spec) {
    String[] rate = physics.feature("auxq").getStringArray("R");
    require(rate.length >= 1, "Missing source dynamic-charge formula");
    physics.feature("auxq").set("R", new String[] {p1ProducerExpression(rate[0])});
    if (IMAGE_REVISION.equals(spec[SPEC_ION_DRAG_REVISION])) {
      physics.feature("idf").set(
          "F", new String[] {IMAGE_FORCE_R, "0[N]", IMAGE_FORCE_Z});
      return;
    }
    String[] ion = physics.feature("idf").getStringArray("F");
    require(ion.length >= 3, "Missing source relative-flow ion-drag formula");
    physics.feature("idf").set(
        "F",
        new String[] {
          p1ProducerExpression(ion[0]), p1ProducerExpression(ion[1]),
          p1ProducerExpression(ion[2])
        });
  }

  private static void configureCustomForce(
      PhysicsFeature feature, String label, String radial, String axial, String study) {
    feature.label(label);
    feature.selection().set(3);
    feature.set("SpecifyForce", "Directly");
    feature.set("F", new String[] {radial, "0[N]", axial});
    feature.set("ParticlesToAffect", "All");
    feature.set("AffectedParticleProperties", "pp1");
    feature.set("StudyStep", study + "/time");
  }

  private static void configurePhysics(Model model, String study, String[] spec) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    for (String tag : new String[] {
        "bf1", "lf1", "auxq", "idf", "ef1", "df1", "thpf1", "liftfm", "depf",
        "gf1", "pp1", "relg1", "wall1", "outin", "outpump", "axi1"
    }) requireFeature(physics, tag);

    physics.feature("bf1").active(false);
    physics.feature("lf1").active(false);
    physics.feature("thpf1").active(false);
    for (String tag : new String[] {"auxq", "idf", "ef1", "df1", "liftfm", "depf", "gf1"}) {
      physics.feature(tag).active(true);
    }

    bindP1Variables(model);
    bindProducerFormulas(physics, spec);
    model.param().set("d0", spec[SPEC_DIAMETER_NM] + "[nm]");
    model.param().set("sigmaR_p", "0.9");

    physics.feature("relg1").set(
        "v0", new String[] {"m3c1_vr0(r,z)", "0[m/s]", "m3c1_vz0(r,z)"});
    physics.feature("relg1").set("aux0_auxq", "m3c1_Z0(r,z)");
    physics.feature("pp1").set("ChargeSpecification", "UserDefined");
    physics.feature("pp1").set("Z", "ZAS");

    physics.feature("ef1").set(
        "E", new String[] {"m3c1_Er(r,z)", "0[V/m]", "m3c1_Ez(r,z)"});
    physics.feature("df1").set(
        "u", new String[] {"m3c1_ugr(r,z)", "0[m/s]", "m3c1_ugz(r,z)"});
    physics.feature("df1").set("rho", "m3c1_rhog(r,z)");
    physics.feature("df1").set("mu", "m3c1_mug(r,z)");
    physics.feature("df1").set("minput_temperature", "m3c1_Tg(r,z)");
    physics.feature("df1").set(
        "pA", "m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]");
    physics.feature("df1").set(
        "minput_pressure",
        "m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]");
    // COMSOL forms the candidate Epstein delta as S + sigmaR*pi/8.
    // S=1 and sigmaR=0.9 therefore yield 1.3534291735288517.
    physics.feature("df1").set("S", "1.0");
    physics.feature("df1").set("sigmaR", "sigmaR_p");
    physics.feature("gf1").set("rho", "m3c1_rhog(r,z)");
    physics.feature("gf1").set("minput_temperature", "m3c1_Tg(r,z)");

    physics.feature("auxq").set("StudyStep", study + "/time");
    physics.feature("idf").set("StudyStep", study + "/time");
    configureCustomForce(
        physics.feature("liftfm"), "M3-C1 common-P1 free-molecular lift", LIFT_R, LIFT_Z, study);
    configureCustomForce(
        physics.feature("depf"), "M3-C1 common-P1 stored-gradient DEP", DEP_R, DEP_Z, study);
    String thermoTag = "m3c1HeatFlux";
    require(!has(physics.feature().tags(), thermoTag), "Common-P1 force tag already exists");
    physics.create(thermoTag, "Force", 2);
    configureCustomForce(
        physics.feature(thermoTag),
        "M3-C1 common-P1 stored-heat-flux Waldmann force",
        THERMO_R,
        THERMO_Z,
        study);

    require(!physics.feature("bf1").isActive(), "Brownian feature remained active");
    require(!physics.feature("lf1").isActive(), "Saffman feature remained active");
    require(!physics.feature("thpf1").isActive(), "Native-gradient thermophoresis remained active");
    for (String tag : new String[] {
        "auxq", "idf", "ef1", "df1", "liftfm", "depf", "gf1", thermoTag,
        "relg1", "wall1", "outin", "outpump", "axi1"
    }) require(physics.feature(tag).isActive(), "Required feature is inactive: " + tag);
    require(
        physics.prop("StoreParticleStatusData").getBoolean("StoreParticleStatusData"),
        "Particle status storage must remain enabled");
    require(
        !physics.prop("StoreExtra").getBoolean("StoreExtra"),
        "Common-P1 run requires StoreExtra=false");
    require(
        "1".equals(physics.prop("WallAccuracyOrder").getString("WallAccuracyOrder")),
        "Common-P1 run requires WallAccuracyOrder=1");
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

  private static String baseSolver(Model model, String study) {
    String[] direct = model.study(study).getSolverSequences("SolverSequence");
    if (direct.length > 0) return direct[0];
    for (String tag : model.study(study).getSolverSequences("All")) {
      try { if (study.equals(model.sol(tag).study())) return tag; }
      catch (Throwable ignored) {}
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

  private static String createAndRunStudy(
      Model model, int stepCode, String profile, String[] spec) {
    String study = "stdM3C1P1" + stepCode;
    String step = stepExpression(stepCode);
    model.param().set("M3C1P1_dt", step);
    require(!has(model.study().tags(), study), "Common-P1 study tag already exists: " + study);
    model.study().create(study);
    model.study(study).label(
        "M3-C1 full-physics canonical-P1 " + spec[SPEC_CASE_ID].replace('_', ' '));

    model.study(study).create("param", "Parametric");
    StudyFeature parameter = model.study(study).feature("param");
    parameter.set("pname", new String[] {"d0"});
    parameter.set("plistarr", new String[] {spec[SPEC_DIAMETER_NM]});
    parameter.set("punit", new String[] {"nm"});
    parameter.set("sweeptype", "sparse");
    try { parameter.set("keepsol", "all"); }
    catch (Throwable ignored) {}

    model.study(study).create("time", "Transient");
    StudyFeature time = model.study(study).feature("time");
    allOff(time, model);
    time.setSolveFor("/physics/" + PHYSICS, true);
    time.set("tlist", timeList(profile));
    time.set("usertol", true);
    time.set("rtol", "1e-8");
    time.set("usesol", true);
    time.set("notsolmethod", "sol");
    time.set("notstudy", BACKGROUND_STUDY);
    time.set("notstudystep", "stat");
    time.set("notsol", BACKGROUND_SOLUTION);
    time.set("notsoluse", "current");
    time.set("notsolnum", "last");

    configurePhysics(model, study, spec);
    model.study(study).createAutoSequences("all");
    String base = baseSolver(model, study);
    SolverFeature transientSolver = model.sol(base).feature("t1");
    transientSolver.set("odesolvertype", "explicit");
    transientSolver.set("timemethodexp", "erk");
    transientSolver.set("erkorder", 4);
    transientSolver.set("rktimestep", "M3C1P1_dt");
    transientSolver.set("rtol", "1e-8");

    long started = System.nanoTime();
    model.study(study).run();
    emit(
        "solve_pass", "step_s", stepSeconds(stepCode), "seconds",
        String.format(Locale.ROOT, "%.3f", (System.nanoTime() - started) / 1e9),
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

  private static String[] force(Physics physics, String tag) {
    String[] values = physics.feature(tag).getStringArray("F");
    require(values.length >= 3, "Expected a three-component force for " + tag);
    return values;
  }

  private static String chargeRate(Physics physics) {
    String[] values = physics.feature("auxq").getStringArray("R");
    require(values.length >= 1, "Missing dynamic-charge rate expression");
    return values[0];
  }

  private static String[] stateExpressions(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    return new String[] {
      "fptas.pidx", "t", "q3r", "q3z", "fptas.vr", "fptas.vz", "ZAS",
      "particlestatus", "fptas.fs", "fptas.st", chargeRate(physics),
      "rho_p*pi*d0^3/6", "m3c1_vr0(q3r,q3z)", "m3c1_vz0(q3r,q3z)",
      "m3c1_Z0(q3r,q3z)"
    };
  }

  private static String[] forceExpressions(Model model) {
    Physics physics = model.component("comp1").physics(PHYSICS);
    String[] ion = force(physics, "idf");
    String[] thermo = force(physics, "m3c1HeatFlux");
    String[] lift = force(physics, "liftfm");
    String[] dep = force(physics, "depf");
    String totalR = "(fptas.ef1.Fer+(" + ion[0] + ")+fptas.df1.FDr+(" + thermo[0]
        + ")+(" + lift[0] + ")+(" + dep[0] + ")+fptas.gf1.Fgr)";
    String totalZ = "(fptas.ef1.Fez+(" + ion[2] + ")+fptas.df1.FDz+(" + thermo[2]
        + ")+(" + lift[2] + ")+(" + dep[2] + ")+fptas.gf1.Fgz)";
    String mass = "(rho_p*pi*d0^3/6)";
    emit("formula", "feature", "auxq", "R", chargeRate(physics));
    emit("formula", "feature", "idf", "F", Arrays.toString(ion));
    emit("formula", "feature", "m3c1HeatFlux", "F", Arrays.toString(thermo));
    emit("formula", "feature", "liftfm", "F", Arrays.toString(lift));
    emit("formula", "feature", "depf", "F", Arrays.toString(dep));
    return new String[] {
      "fptas.pidx", "t", "fptas.ef1.Fer", "fptas.ef1.Fez", ion[0], ion[2],
      "fptas.df1.FDr", "fptas.df1.FDz", thermo[0], thermo[2], lift[0], lift[2],
      dep[0], dep[2], "fptas.gf1.Fgr", "fptas.gf1.Fgz", totalR, totalZ,
      totalR + "/" + mass, totalZ + "/" + mass
    };
  }

  private static String[] primitiveExpressions() {
    String[] values = new String[P1_NAMES.length + 2];
    values[0] = "fptas.pidx";
    values[1] = "t";
    for (int index = 0; index < P1_NAMES.length; index++) {
      values[index + 2] = P1_NAMES[index] + "(q3r,q3z)";
    }
    return values;
  }

  private static void exportTable(
      Model model,
      String dataset,
      String directory,
      String tag,
      String filename,
      String[] expressions,
      String[] units,
      String[] columns) {
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
    try { export.set("outerinput", "all"); }
    catch (Throwable ignored) {}
    export.run();
    emit("export_pass", "table", filename);
  }

  private static void exportHistory(Model model, String dataset, String directory) {
    exportTable(
        model, dataset, directory, "m3c1P1State", "state_raw_wide.csv",
        stateExpressions(model), STATE_UNITS, STATE_COLUMNS);
    exportTable(
        model, dataset, directory, "m3c1P1Force", "force_raw_wide.csv",
        forceExpressions(model), FORCE_UNITS, FORCE_COLUMNS);
    exportTable(
        model, dataset, directory, "m3c1P1Primitive", "primitive_raw_wide.csv",
        primitiveExpressions(), PRIMITIVE_UNITS, PRIMITIVE_COLUMNS);
  }

  private static int particleRows(Model model, String dataset) {
    String tag = "m3c1P1ParticleCount";
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
      try { model.result().numerical().remove(tag); }
      catch (Throwable ignored) {}
    }
  }

  private static String deterministicContributions(String[] spec) {
    return "electric," + spec[SPEC_CONTRIBUTION]
        + ",epstein_drag,waldmann_heat_flux_thermophoresis,"
        + "free_molecular_lift_sensitivity,dielectrophoresis,gravity_buoyancy";
  }

  private static void runOne(int stepCode, String profile, String[] spec) throws Exception {
    Model model = null;
    String directory = stepDirectory(stepCode);
    try {
      model = ModelUtil.loadCopy(
          "M3C1CommonP1" + spec[SPEC_DIAMETER_NM] + "nm" + stepCode, SOURCE);
      createSectionwiseFunctions(model);
      createReleaseFunctions(model);
      String solution = createAndRunStudy(model, stepCode, profile, spec);
      String dataset = "partM3C1P1" + stepCode;
      createParticleDataset(model, dataset, solution);
      double[] times = model.sol(solution).getPVals();
      require(
          times.length == expectedOutputTimes(profile),
          "Expected " + expectedOutputTimes(profile) + " output times, got " + times.length);
      require(Math.abs(times[0]) < 1e-15, "Unexpected first output time");
      require(
          Math.abs(times[times.length - 1] - expectedEndTime(profile)) < 1e-14,
          "Unexpected final time");
      int particles = particleRows(model, dataset);
      require(particles == 287, "Expected 287 particles, got " + particles);
      exportHistory(model, dataset, directory);
      emit(
          "configuration", "run_profile", profile, "step_s", stepSeconds(stepCode),
          "time_start_s", "0.0", "time_end_s",
          String.format(Locale.ROOT, "%.8g", expectedEndTime(profile)),
          "brownian_active", "false",
          "saffman_active", "false", "dynamic_charge_active", "true",
          "native_thermophoresis_active", "false", "common_heat_flux_force_active", "true",
          "field_source", "canonical_exact_connectivity_P1_sectionwise",
          "initial_state_source", "candidate_realized_source_table",
          "primitive_function_count", Integer.toString(P1_NAMES.length),
          "deterministic_contributions", deterministicContributions(spec),
          "integrator", "classical_rk4", "integrator_order", "4",
          "relative_tolerance", "1e-8", "wall_accuracy_order", "1",
          "store_particle_status", "true", "store_extra", "false", "physics", PHYSICS,
          "study", "stdM3C1P1" + stepCode, "solution", solution,
          "output_times", Integer.toString(times.length), "particle_rows", Integer.toString(particles),
          "source_model", SOURCE, "model_saved", "false");
    } finally {
      if (model != null) ModelUtil.remove(model.tag());
      System.gc();
    }
  }

  private static void run(String profile) throws Exception {
    run(profile, legacy100Relative(), false);
  }

  private static void run(String profile, String[] spec, boolean campaign) throws Exception {
    try {
      require(
          PRE_EVENT_PROFILE.equals(profile) || MATERIAL_EVENT_PROFILE.equals(profile),
          "Unsupported M3-C1 common-P1 run profile: " + profile);
      System.out.println("M3C1_COMMON_P1|launch|run_profile=" + profile);
      System.out.flush();
      ModelUtil.showProgress(false);
      if (campaign) {
        emit(
            "campaign_spec", "case_id", spec[SPEC_CASE_ID],
            "diameter_nm", spec[SPEC_DIAMETER_NM],
            "ion_drag_revision", spec[SPEC_ION_DRAG_REVISION],
            "deterministic_contribution_name", spec[SPEC_CONTRIBUTION]);
      }
      for (int stepCode : stepCodes(profile)) runOne(stepCode, profile, spec);
      emit(
          "run_pass", "case", spec[SPEC_CASE_ID], "run_profile", profile, "steps",
          isMaterialEventProfile(profile) ? "1.5625e-7" : "6.25e-7,3.125e-7,1.5625e-7",
          "common_field", "canonical_exact_connectivity_P1_sectionwise",
          "model_saved", "false");
    } catch (Throwable failure) {
      System.out.println(
          "M3C1_COMMON_P1|fatal|exception=" + failure.getClass().getName());
      failure.printStackTrace(System.out);
      System.out.flush();
      if (failure instanceof Error) throw (Error) failure;
      if (failure instanceof Exception) throw (Exception) failure;
      throw new RuntimeException(failure);
    }
  }

  static void runMaterialEvent() throws Exception {
    run(MATERIAL_EVENT_PROFILE);
  }

  static void runCampaign(
      String caseId, int diameterNm, String revision, String contribution) throws Exception {
    run(
        PRE_EVENT_PROFILE,
        campaignSpec(caseId, diameterNm, revision, contribution),
        true);
  }

  public static void main(String[] args) throws Exception {
    run(PRE_EVENT_PROFILE);
  }
}

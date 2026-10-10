import com.comsol.model.Model;

/** One post-interpolation Epstein coefficient for the C2 and C3 companions. */
public final class CommonP1Epstein {
  private CommonP1Epstein() {}

  // The registered effective-gas campaign parameters. Readback records their
  // resolved SI values; the meaning preflight must bind them to the candidate.
  static final String MOLECULAR_MASS = "1.2753471408396638e-25[kg]";
  static final String DELTA = "1.3534291735288517";
  static final String VARIABLE_TAG = "m3cEpstein";

  static String coefficient() {
    return "(4*pi/3)*(d0/2)^2*m3c1_rhog(r,z)"
        + "*sqrt(8*k_B_const*m3c1_Tg(r,z)/(pi*m3c_mgas))*m3c_delta";
  }

  static String[] force(String physics) {
    return new String[] {
      "m3c_beta*(m3c1_ugr(r,z)-" + physics + ".vr)",
      "0[N]",
      "m3c_beta*(m3c1_ugz(r,z)-" + physics + ".vz)"
    };
  }

  static void bind(Model model) {
    model.param().set("m3c_mgas", MOLECULAR_MASS);
    model.param().set("m3c_delta", DELTA);
    model.component("comp1").variable().create(VARIABLE_TAG);
    model.component("comp1").variable(VARIABLE_TAG).set("m3c_beta", coefficient());
    model.component("comp1").variable(VARIABLE_TAG).set("m3c_muB", "m3c_beta/(3*pi*d0)");
  }
}

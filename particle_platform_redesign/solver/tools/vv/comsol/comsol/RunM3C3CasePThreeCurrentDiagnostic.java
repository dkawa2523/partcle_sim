/**
 * External-V&amp;V-only entry point for the M3-C3 Case-P 100 nm force discrepancy.
 *
 * <p>The shared runner owns the physics configuration and reloads {@code source_copy.mph} with
 * {@code ModelUtil.loadCopy}. This entry point fixes the rerun to 1.25 us and enables only the
 * additional state, force-component, and charge-rate exports.
 */
public final class RunM3C3CasePThreeCurrentDiagnostic {
  private RunM3C3CasePThreeCurrentDiagnostic() {}

  public static void main(String[] args) throws Exception {
    RunM3C3CasePThreeCurrent.runFineDiagnostic();
  }
}

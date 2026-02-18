import PathWrapper from "../components/PathWrapper";
import { BUSINESS_PATH } from "../data/data";

export function BusinessPath() {
  return <PathWrapper root={BUSINESS_PATH[0]} />;
}

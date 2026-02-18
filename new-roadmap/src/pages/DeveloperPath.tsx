import PathWrapper from "../components/PathWrapper";
import { DEV_PATH } from "../data/data";

export default function DeveloperPath() {
  return <PathWrapper root={DEV_PATH[0]} />;
}
